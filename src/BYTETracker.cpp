#include "ByteTrack/BYTETracker.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

byte_track::BYTETracker::BYTETracker(const ByteTrackerConfig &config) :
    config_(config),
    max_time_lost_(static_cast<size_t>(config.track_buffer)),
    frame_id_(0),
    track_id_count_(0),
    expected_dt_ms_(1000.0f / config.frame_rate),
    last_timestamp_ms_(-1),
    current_dt_(1.0f),
    has_timestamp_(false)
{
}

byte_track::BYTETracker::~BYTETracker()
{
}

std::vector<byte_track::BYTETracker::STrackPtr> byte_track::BYTETracker::getAllActiveTracks() const
{
    std::vector<STrackPtr> all_tracks;
    all_tracks.reserve(tracked_stracks_.size() + lost_stracks_.size());

    for (const auto &track : tracked_stracks_)
    {
        all_tracks.push_back(track);
    }

    for (const auto &track : lost_stracks_)
    {
        all_tracks.push_back(track);
    }

    return all_tracks;
}

std::vector<byte_track::BYTETracker::STrackPtr> byte_track::BYTETracker::update(const std::vector<Object>& objects)
{
    return update(objects, -1);
}

std::vector<byte_track::BYTETracker::STrackPtr> byte_track::BYTETracker::update(const std::vector<Object>& objects,
                                                                                int64_t timestamp_ms)
{
    ////////////////// Step 1: Get detections //////////////////
    frame_id_++;

    if (timestamp_ms >= 0)
    {
        has_timestamp_ = true;
        if (last_timestamp_ms_ >= 0)
        {
            int64_t dt_ms = timestamp_ms - last_timestamp_ms_;
            if (dt_ms > 0)
            {
                current_dt_ = static_cast<float>(dt_ms) / expected_dt_ms_;
            }
            else
            {
                current_dt_ = 1.0f;
            }
        }
        last_timestamp_ms_ = timestamp_ms;
    }
    else
    {
        current_dt_ = 1.0f;
    }

    // Create new STracks using the result of object detection
    std::vector<STrackPtr> det_stracks;
    std::vector<STrackPtr> det_low_stracks;

    for (const auto &object : objects)
    {
        const auto strack = std::make_shared<STrack>(object.rect, object.prob);
        if (object.prob >= config_.track_thresh)
        {
            det_stracks.push_back(strack);
        }
        else
        {
            det_low_stracks.push_back(strack);
        }
    }

    // Create lists of existing STrack
    std::vector<STrackPtr> active_stracks;
    std::vector<STrackPtr> non_active_stracks;
    std::vector<STrackPtr> strack_pool;

    for (const auto& tracked_strack : tracked_stracks_)
    {
        if (!tracked_strack->isActivated())
        {
            non_active_stracks.push_back(tracked_strack);
        }
        else
        {
            active_stracks.push_back(tracked_strack);
        }
    }

    strack_pool = jointStracks(active_stracks, lost_stracks_);

    // Predict current pose by KF
    for (auto &strack : strack_pool)
    {
        strack->predict(current_dt_, config_.tracked_predict_dt_cap, config_.lost_predict_dt_cap);
    }

    ////////////////// Step 2: First association, with IoU //////////////////
    std::vector<STrackPtr> current_tracked_stracks;
    std::vector<STrackPtr> remain_tracked_stracks;
    std::vector<STrackPtr> remain_det_stracks;
    std::vector<STrackPtr> refind_stracks;

    {
        std::vector<std::vector<int>> matches_idx;
        std::vector<int> unmatch_detection_idx, unmatch_track_idx;

        const auto dists = calcIouDistance(strack_pool, det_stracks, config_.lost_center_match_scale);
        linearAssignment(dists, strack_pool.size(), det_stracks.size(), config_.match_thresh,
                         matches_idx, unmatch_track_idx, unmatch_detection_idx);

        for (const auto &match_idx : matches_idx)
        {
            const auto track = strack_pool[match_idx[0]];
            const auto det = det_stracks[match_idx[1]];
            if (track->getSTrackState() == STrackState::Tracked)
            {
                track->update(*det, frame_id_, config_.vel_ema_alpha);
                current_tracked_stracks.push_back(track);
            }
            else
            {
                track->reActivate(*det, frame_id_, -1, config_.vel_ema_alpha);
                refind_stracks.push_back(track);
            }
            track->setLastSeenTimestampMs(timestamp_ms);
        }

        for (const auto &unmatch_idx : unmatch_detection_idx)
        {
            remain_det_stracks.push_back(det_stracks[unmatch_idx]);
        }

        for (const auto &unmatch_idx : unmatch_track_idx)
        {
            if (strack_pool[unmatch_idx]->getSTrackState() == STrackState::Tracked)
            {
                remain_tracked_stracks.push_back(strack_pool[unmatch_idx]);
            }
        }
    }

    ////////////////// Step 3: Second association, using low score dets //////////////////
    std::vector<STrackPtr> current_lost_stracks;

    {
        std::vector<std::vector<int>> matches_idx;
        std::vector<int> unmatch_track_idx, unmatch_detection_idx;

        const auto dists = calcIouDistance(remain_tracked_stracks, det_low_stracks);
        linearAssignment(dists, remain_tracked_stracks.size(), det_low_stracks.size(), config_.low_score_match_thresh,
                         matches_idx, unmatch_track_idx, unmatch_detection_idx);

        for (const auto &match_idx : matches_idx)
        {
            const auto track = remain_tracked_stracks[match_idx[0]];
            const auto det = det_low_stracks[match_idx[1]];
            if (track->getSTrackState() == STrackState::Tracked)
            {
                track->update(*det, frame_id_, config_.vel_ema_alpha);
                current_tracked_stracks.push_back(track);
            }
            else
            {
                track->reActivate(*det, frame_id_, -1, config_.vel_ema_alpha);
                refind_stracks.push_back(track);
            }
            track->setLastSeenTimestampMs(timestamp_ms);
        }

        for (const auto &unmatch_track : unmatch_track_idx)
        {
            const auto track = remain_tracked_stracks[unmatch_track];
            if (track->getSTrackState() != STrackState::Lost)
            {
                track->markAsLost();
                current_lost_stracks.push_back(track);
            }
        }
    }

    ////////////////// Step 4: Init new stracks //////////////////
    std::vector<STrackPtr> current_removed_stracks;

    {
        std::vector<int> unmatch_detection_idx;
        std::vector<int> unmatch_unconfirmed_idx;
        std::vector<std::vector<int>> matches_idx;

        // Deal with unconfirmed tracks, usually tracks with only one beginning frame
        const auto dists = calcIouDistance(non_active_stracks, remain_det_stracks);
        linearAssignment(dists, non_active_stracks.size(), remain_det_stracks.size(), config_.unconfirmed_match_thresh,
                         matches_idx, unmatch_unconfirmed_idx, unmatch_detection_idx);

        for (const auto &match_idx : matches_idx)
        {
            non_active_stracks[match_idx[0]]->update(*remain_det_stracks[match_idx[1]],
                                                    frame_id_, config_.vel_ema_alpha);
            non_active_stracks[match_idx[0]]->setLastSeenTimestampMs(timestamp_ms);
            current_tracked_stracks.push_back(non_active_stracks[match_idx[0]]);
        }

        for (const auto &unmatch_idx : unmatch_unconfirmed_idx)
        {
            const auto track = non_active_stracks[unmatch_idx];
            track->markAsRemoved();
            current_removed_stracks.push_back(track);
        }

        // Add new stracks
        for (const auto &unmatch_idx : unmatch_detection_idx)
        {
            const auto track = remain_det_stracks[unmatch_idx];
            if (track->getScore() < config_.high_thresh)
            {
                continue;
            }
            track_id_count_++;
            track->activate(frame_id_, track_id_count_);
            track->setLastSeenTimestampMs(timestamp_ms);
            current_tracked_stracks.push_back(track);
        }
    }

    ////////////////// Step 5: Update state //////////////////
    const bool use_time_based_expiry =
        config_.track_buffer_ms > 0 && has_timestamp_ && timestamp_ms >= 0;

    for (const auto &lost_strack : lost_stracks_)
    {
        bool expired;
        if (use_time_based_expiry && lost_strack->getLastSeenTimestampMs() >= 0)
        {
            expired = (timestamp_ms - lost_strack->getLastSeenTimestampMs())
                      > static_cast<int64_t>(config_.track_buffer_ms);
        }
        else
        {
            expired = (frame_id_ - lost_strack->getFrameId()) > max_time_lost_; //max_time_lost_ in this case is derived from the frame_id and assuming constant frame rate
        }

        if (expired)
        {
            lost_strack->markAsRemoved();
            current_removed_stracks.push_back(lost_strack);
        }
    }

    tracked_stracks_ = jointStracks(current_tracked_stracks, refind_stracks);
    lost_stracks_ = subStracks(jointStracks(subStracks(lost_stracks_, tracked_stracks_), current_lost_stracks), removed_stracks_);
    removed_stracks_ = jointStracks(removed_stracks_, current_removed_stracks);

    std::vector<STrackPtr> tracked_stracks_out, lost_stracks_out;
    removeDuplicateStracks(tracked_stracks_, lost_stracks_, tracked_stracks_out, lost_stracks_out);
    tracked_stracks_ = tracked_stracks_out;
    lost_stracks_ = lost_stracks_out;

    std::vector<STrackPtr> output_stracks;
    for (const auto &track : tracked_stracks_)
    {
        if (track->isActivated())
        {
            output_stracks.push_back(track);
        }
    }

    return output_stracks;
}
std::vector<byte_track::BYTETracker::STrackPtr> byte_track::BYTETracker::jointStracks(const std::vector<STrackPtr> &a_tlist,
                                                                                      const std::vector<STrackPtr> &b_tlist) const
{
    std::map<int, int> exists;
    std::vector<STrackPtr> res;
    for (size_t i = 0; i < a_tlist.size(); i++)
    {
        exists.emplace(a_tlist[i]->getTrackId(), 1);
        res.push_back(a_tlist[i]);
    }
    for (size_t i = 0; i < b_tlist.size(); i++)
    {
        const int &tid = b_tlist[i]->getTrackId();
        if (!exists[tid] || exists.count(tid) == 0)
        {
            exists[tid] = 1;
            res.push_back(b_tlist[i]);
        }
    }
    return res;
}

std::vector<byte_track::BYTETracker::STrackPtr> byte_track::BYTETracker::subStracks(const std::vector<STrackPtr> &a_tlist,
                                                                                    const std::vector<STrackPtr> &b_tlist) const
{
    std::map<int, STrackPtr> stracks;
    for (size_t i = 0; i < a_tlist.size(); i++)
    {
        stracks.emplace(a_tlist[i]->getTrackId(), a_tlist[i]);
    }

    for (size_t i = 0; i < b_tlist.size(); i++)
    {
        const int &tid = b_tlist[i]->getTrackId();
        if (stracks.count(tid) != 0)
        {
            stracks.erase(tid);
        }
    }

    std::vector<STrackPtr> res;
    std::map<int, STrackPtr>::iterator it;
    for (it = stracks.begin(); it != stracks.end(); ++it)
    {
        res.push_back(it->second);
    }

    return res;
}

void byte_track::BYTETracker::removeDuplicateStracks(const std::vector<STrackPtr> &a_stracks,
                                                     const std::vector<STrackPtr> &b_stracks,
                                                     std::vector<STrackPtr> &a_res,
                                                     std::vector<STrackPtr> &b_res) const
{
    const auto ious = calcIouDistance(a_stracks, b_stracks);

    std::vector<std::pair<size_t, size_t>> overlapping_combinations;
    for (size_t i = 0; i < ious.size(); i++)
    {
        for (size_t j = 0; j < ious[i].size(); j++)
        {
            if (ious[i][j] < 0.15)
            {
                overlapping_combinations.emplace_back(i, j);
            }
        }
    }

    std::vector<bool> a_overlapping(a_stracks.size(), false), b_overlapping(b_stracks.size(), false);
    for (const auto &[a_idx, b_idx] : overlapping_combinations)
    {
        const int timep = a_stracks[a_idx]->getFrameId() - a_stracks[a_idx]->getStartFrameId();
        const int timeq = b_stracks[b_idx]->getFrameId() - b_stracks[b_idx]->getStartFrameId();
        if (timep > timeq)
        {
            b_overlapping[b_idx] = true;
        }
        else
        {
            a_overlapping[a_idx] = true;
        }
    }

    for (size_t ai = 0; ai < a_stracks.size(); ai++)
    {
        if (!a_overlapping[ai])
        {
            a_res.push_back(a_stracks[ai]);
        }
    }

    for (size_t bi = 0; bi < b_stracks.size(); bi++)
    {
        if (!b_overlapping[bi])
        {
            b_res.push_back(b_stracks[bi]);
        }
    }
}

void byte_track::BYTETracker::linearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                                               const int &cost_matrix_size,
                                               const int &cost_matrix_size_size,
                                               const float &thresh,
                                               std::vector<std::vector<int>> &matches,
                                               std::vector<int> &a_unmatched,
                                               std::vector<int> &b_unmatched) const
{
    if (cost_matrix.size() == 0)
    {
        for (int i = 0; i < cost_matrix_size; i++)
        {
            a_unmatched.push_back(i);
        }
        for (int i = 0; i < cost_matrix_size_size; i++)
        {
            b_unmatched.push_back(i);
        }
        return;
    }

    std::vector<int> rowsol; std::vector<int> colsol;
    execLapjv(cost_matrix, rowsol, colsol, true, thresh);
    for (size_t i = 0; i < rowsol.size(); i++)
    {
        if (rowsol[i] >= 0)
        {
            std::vector<int> match;
            match.push_back(i);
            match.push_back(rowsol[i]);
            matches.push_back(match);
        }
        else
        {
            a_unmatched.push_back(i);
        }
    }

    for (size_t i = 0; i < colsol.size(); i++)
    {
        if (colsol[i] < 0)
        {
            b_unmatched.push_back(i);
        }
    }
}

std::vector<std::vector<float>> byte_track::BYTETracker::calcIous(const std::vector<Rect<float>> &a_rect,
                                                                  const std::vector<Rect<float>> &b_rect) const
{
    std::vector<std::vector<float>> ious;
    if (a_rect.size() * b_rect.size() == 0)
    {
        return ious;
    }

    ious.resize(a_rect.size());
    for (size_t i = 0; i < ious.size(); i++)
    {
        ious[i].resize(b_rect.size());
    }

    for (size_t bi = 0; bi < b_rect.size(); bi++)
    {
        for (size_t ai = 0; ai < a_rect.size(); ai++)
        {
            ious[ai][bi] = b_rect[bi].calcIoU(a_rect[ai]);
        }
    }
    return ious;
}

float byte_track::BYTETracker::calcCenterSimilarity(const Rect<float> &a_rect,
                                                    const Rect<float> &b_rect,
                                                    const float scale) const
{
    const float a_h = std::max(a_rect.height(), 1.0f);
    const float b_h = std::max(b_rect.height(), 1.0f);
    const float dist_scale = scale * (a_h + b_h) * 0.5f;
    if (dist_scale <= 0.0f)
    {
        return 0.0f;
    }

    const float dx = (a_rect.x() + a_rect.width() * 0.5f) - (b_rect.x() + b_rect.width() * 0.5f);
    const float dy = (a_rect.y() + a_rect.height() * 0.5f) - (b_rect.y() + b_rect.height() * 0.5f);
    const float dist = std::sqrt(dx * dx + dy * dy);

    return std::max(0.0f, 1.0f - dist / dist_scale);
}

std::vector<std::vector<float> > byte_track::BYTETracker::calcIouDistance(const std::vector<STrackPtr> &a_tracks,
                                                                          const std::vector<STrackPtr> &b_tracks,
                                                                          const float lost_center_match_scale) const
{
    std::vector<byte_track::Rect<float>> a_rects, b_rects;
    for (size_t i = 0; i < a_tracks.size(); i++)
    {
        a_rects.push_back(a_tracks[i]->getRect());
    }

    for (size_t i = 0; i < b_tracks.size(); i++)
    {
        b_rects.push_back(b_tracks[i]->getRect());
    }

    const auto ious = calcIous(a_rects, b_rects);

    // Lost tracks whose Kalman prediction has drifted off-screen produce
    // near-zero IoU even against the correct detection. Allow them to still
    // associate by center distance so the same id can be re-acquired.
    constexpr float MIN_IOU_FOR_CENTER_MATCH = 0.05f;

    std::vector<std::vector<float>> cost_matrix;
    for (size_t i = 0; i < ious.size(); i++)
    {
        const bool use_center_match = lost_center_match_scale > 0.0f &&
                                      a_tracks[i]->getSTrackState() == STrackState::Lost;

        std::vector<float> iou;
        for (size_t j = 0; j < ious[i].size(); j++)
        {
            float similarity = ious[i][j];
            if (use_center_match && similarity < MIN_IOU_FOR_CENTER_MATCH)
            {
                similarity = std::max(similarity,
                                      calcCenterSimilarity(a_rects[i], b_rects[j], lost_center_match_scale));
            }

            float cost = 1 - similarity;

            // A re-entering or clipped object legitimately changes apparent size, so the
            // area-ratio penalty would wrongly reject lost-track re-association. Skip it for
            // lost tracks; center-distance gating already constrains them spatially.
            if (!use_center_match)
            {
                float area_a = a_rects[i].width() * a_rects[i].height();
                float area_b = b_rects[j].width() * b_rects[j].height();
                if (area_a > 0 && area_b > 0)
                {
                    float ratio = std::max(area_a, area_b) / std::min(area_a, area_b);
                    if (ratio > config_.max_area_ratio)
                    {
                        cost = 1.0f;
                    }
                }
            }

            iou.push_back(cost);
        }
        cost_matrix.push_back(iou);
    }

    return cost_matrix;
}

double byte_track::BYTETracker::execLapjv(const std::vector<std::vector<float>> &cost,
                                          std::vector<int> &rowsol,
                                          std::vector<int> &colsol,
                                          bool extend_cost,
                                          float cost_limit,
                                          bool return_cost) const
{
    std::vector<std::vector<float> > cost_c;
    cost_c.assign(cost.begin(), cost.end());

    std::vector<std::vector<float> > cost_c_extended;

    int n_rows = cost.size();
    int n_cols = cost[0].size();
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols)
    {
        n = n_rows;
    }
    else
    {
        if (!extend_cost)
        {
            throw std::runtime_error("The `extend_cost` variable should set True");
        }
    }

    if (extend_cost || cost_limit < std::numeric_limits<float>::max())
    {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (size_t i = 0; i < cost_c_extended.size(); i++)
            cost_c_extended[i].resize(n);

        if (cost_limit < std::numeric_limits<float>::max())
        {
            for (size_t i = 0; i < cost_c_extended.size(); i++)
            {
                for (size_t j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_limit / 2.0;
                }
            }
        }
        else
        {
            float cost_max = -1;
            for (size_t i = 0; i < cost_c.size(); i++)
            {
                for (size_t j = 0; j < cost_c[i].size(); j++)
                {
                    if (cost_c[i][j] > cost_max)
                        cost_max = cost_c[i][j];
                }
            }
            for (size_t i = 0; i < cost_c_extended.size(); i++)
            {
                for (size_t j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_max + 1;
                }
            }
        }

        for (size_t i = n_rows; i < cost_c_extended.size(); i++)
        {
            for (size_t j = n_cols; j < cost_c_extended[i].size(); j++)
            {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    double **cost_ptr;
    cost_ptr = new double *[sizeof(double *) * n];
    for (int i = 0; i < n; i++)
        cost_ptr[i] = new double[sizeof(double) * n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cost_ptr[i][j] = cost_c[i][j];
        }
    }

    int* x_c = new int[sizeof(int) * n];
    int *y_c = new int[sizeof(int) * n];

    int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
    if (ret != 0)
    {
        throw std::runtime_error("The result of lapjv_internal() is invalid.");
    }

    double opt = 0.0;

    if (n != n_rows)
    {
        for (int i = 0; i < n; i++)
        {
            if (x_c[i] >= n_cols)
                x_c[i] = -1;
            if (y_c[i] >= n_rows)
                y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; i++)
        {
            rowsol[i] = x_c[i];
        }
        for (int i = 0; i < n_cols; i++)
        {
            colsol[i] = y_c[i];
        }

        if (return_cost)
        {
            for (size_t i = 0; i < rowsol.size(); i++)
            {
                if (rowsol[i] != -1)
                {
                    opt += cost_ptr[i][rowsol[i]];
                }
            }
        }
    }
    else if (return_cost)
    {
        for (size_t i = 0; i < rowsol.size(); i++)
        {
            opt += cost_ptr[i][rowsol[i]];
        }
    }

    for (int i = 0; i < n; i++)
    {
        delete[]cost_ptr[i];
    }
    delete[]cost_ptr;
    delete[]x_c;
    delete[]y_c;

    return opt;
}
