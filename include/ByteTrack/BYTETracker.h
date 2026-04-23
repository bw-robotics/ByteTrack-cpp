#pragma once

#include "ByteTrack/STrack.h"
#include "ByteTrack/lapjv.h"
#include "ByteTrack/Object.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace byte_track
{

struct ByteTrackerConfig
{
    int frame_rate = 10;
    int track_buffer = 20;
    float track_thresh = 0.5f;
    float high_thresh = 0.6f;
    float match_thresh = 0.8f;
    float max_area_ratio = 3.0f;
    float low_score_match_thresh = 0.5f;
    float unconfirmed_match_thresh = 0.7f;
    float tracked_predict_dt_cap = 3.0f;
    float lost_predict_dt_cap = 2.0f;

    std::string asString() const
    {
        return "frame_rate=" + std::to_string(frame_rate) +
               ", track_buffer=" + std::to_string(track_buffer) +
               ", track_thresh=" + std::to_string(track_thresh) +
               ", high_thresh=" + std::to_string(high_thresh) +
               ", match_thresh=" + std::to_string(match_thresh) +
               ", max_area_ratio=" + std::to_string(max_area_ratio) +
               ", low_score_match_thresh=" + std::to_string(low_score_match_thresh) +
               ", unconfirmed_match_thresh=" + std::to_string(unconfirmed_match_thresh) +
               ", tracked_predict_dt_cap=" + std::to_string(tracked_predict_dt_cap) +
               ", lost_predict_dt_cap=" + std::to_string(lost_predict_dt_cap);
    }
};

class BYTETracker
{
public:
    using STrackPtr = std::shared_ptr<STrack>;

    explicit BYTETracker(const ByteTrackerConfig &config = {});
    ~BYTETracker();

    std::vector<STrackPtr> update(const std::vector<Object>& objects);
    std::vector<STrackPtr> update(const std::vector<Object>& objects, int64_t timestamp_ms);

    float getCurrentDt() const { return current_dt_; }
    float getExpectedDtMs() const { return expected_dt_ms_; }
    const std::vector<STrackPtr> &getLostTracks() const { return lost_stracks_; }
    std::vector<STrackPtr> getAllActiveTracks() const;

private:
    std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr> &a_tlist,
                                        const std::vector<STrackPtr> &b_tlist) const;

    std::vector<STrackPtr> subStracks(const std::vector<STrackPtr> &a_tlist,
                                      const std::vector<STrackPtr> &b_tlist) const;

    void removeDuplicateStracks(const std::vector<STrackPtr> &a_stracks,
                                const std::vector<STrackPtr> &b_stracks,
                                std::vector<STrackPtr> &a_res,
                                std::vector<STrackPtr> &b_res) const;

    void linearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                          const int &cost_matrix_size,
                          const int &cost_matrix_size_size,
                          const float &thresh,
                          std::vector<std::vector<int>> &matches,
                          std::vector<int> &b_unmatched,
                          std::vector<int> &a_unmatched) const;

    std::vector<std::vector<float>> calcIouDistance(const std::vector<STrackPtr> &a_tracks,
                                                    const std::vector<STrackPtr> &b_tracks) const;

    std::vector<std::vector<float>> calcIous(const std::vector<Rect<float>> &a_rect,
                                             const std::vector<Rect<float>> &b_rect) const;

    double execLapjv(const std::vector<std::vector<float> > &cost,
                     std::vector<int> &rowsol,
                     std::vector<int> &colsol,
                     bool extend_cost = false,
                     float cost_limit = std::numeric_limits<float>::max(),
                     bool return_cost = true) const;

private:
    const ByteTrackerConfig config_;
    const size_t max_time_lost_;

    size_t frame_id_;
    size_t track_id_count_;

    float expected_dt_ms_;
    int64_t last_timestamp_ms_;
    float current_dt_;
    bool has_timestamp_;

    std::vector<STrackPtr> tracked_stracks_;
    std::vector<STrackPtr> lost_stracks_;
    std::vector<STrackPtr> removed_stracks_;
};
}
