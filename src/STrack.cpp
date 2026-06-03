#include "ByteTrack/STrack.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

byte_track::STrack::STrack(const Rect<float>& rect, const float& score) :
    kalman_filter_(),
    mean_(),
    covariance_(),
    rect_(rect),
    state_(STrackState::New),
    is_activated_(false),
    score_(score),
    track_id_(0),
    frame_id_(0),
    start_frame_id_(0),
    tracklet_len_(0)
{
}

byte_track::STrack::~STrack()
{
}

const byte_track::Rect<float>& byte_track::STrack::getRect() const
{
    return rect_;
}

const byte_track::STrackState& byte_track::STrack::getSTrackState() const
{
    return state_;
}

const bool& byte_track::STrack::isActivated() const
{
    return is_activated_;
}
const float& byte_track::STrack::getScore() const
{
    return score_;
}

const size_t& byte_track::STrack::getTrackId() const
{
    return track_id_;
}

const size_t& byte_track::STrack::getFrameId() const
{
    return frame_id_;
}

const size_t& byte_track::STrack::getStartFrameId() const
{
    return start_frame_id_;
}

const size_t& byte_track::STrack::getTrackletLength() const
{
    return tracklet_len_;
}

void byte_track::STrack::activate(const size_t& frame_id, const size_t& track_id)
{
    kalman_filter_.initiate(mean_, covariance_, rect_.getXyah());

    updateRect();

    state_ = STrackState::Tracked;
    if (frame_id == 1)
    {
        is_activated_ = true;
    }
    track_id_ = track_id;
    frame_id_ = frame_id;
    start_frame_id_ = frame_id;
    tracklet_len_ = 0;
}

void byte_track::STrack::reActivate(const STrack &new_track, const size_t &frame_id,
                                    const int &new_track_id, float vel_ema_alpha)
{
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());
    applyVelocityEma(vel_ema_alpha);

    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    if (0 <= new_track_id)
    {
        track_id_ = new_track_id;
    }
    frame_id_ = frame_id;
    tracklet_len_ = 0;
}

void byte_track::STrack::predict(float dt, float tracked_dt_cap, float lost_dt_cap)
{
    const float per_step_cap = (state_ == STrackState::Tracked) ? tracked_dt_cap : lost_dt_cap;

    if (state_ != STrackState::Tracked)
    {
        mean_[7] = 0;
    }

    // Advance by the full elapsed time using capped sub-steps. A sparse detector
    // cadence yields a single large-dt update; clamping it (min(dt, cap)) would
    // under-predict a lost track after a gap, so instead we accumulate the motion
    // in cap-sized steps (equivalent to frame-by-frame stepping, bounded by the
    // track lifetime).
    float remaining = dt;
    const float step_cap = per_step_cap > 0.0f ? per_step_cap : dt;
    do
    {
        const float step = (step_cap > 0.0f) ? std::min(remaining, step_cap) : remaining;
        kalman_filter_.predict(mean_, covariance_, step);
        remaining -= step;
    } while (remaining > 1e-3f && step_cap > 0.0f);

    updateRect();
}

void byte_track::STrack::update(const STrack &new_track, const size_t &frame_id, float vel_ema_alpha)
{
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());
    applyVelocityEma(vel_ema_alpha);

    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    frame_id_ = frame_id;
    tracklet_len_++;
}

std::pair<float, float> byte_track::STrack::getVelocity() const
{
    return std::make_pair(mean_[4], mean_[5]);
}

void byte_track::STrack::markAsLost()
{
    state_ = STrackState::Lost;
}

void byte_track::STrack::markAsRemoved()
{
    state_ = STrackState::Removed;
}

void byte_track::STrack::updateRect()
{
    rect_.width() = mean_[2] * mean_[3];
    rect_.height() = mean_[3];
    rect_.x() = mean_[0] - rect_.width() / 2;
    rect_.y() = mean_[1] - rect_.height() / 2;
}

void byte_track::STrack::applyVelocityEma(float alpha)
{
    if (alpha >= 1.0f)
    {
        return;
    }
    const float kf_vx = mean_[4];
    const float kf_vy = mean_[5];
    if (!ema_initialized_)
    {
        ema_vel_x_ = kf_vx;
        ema_vel_y_ = kf_vy;
        ema_initialized_ = true;
        return;
    }
    ema_vel_x_ = alpha * kf_vx + (1.0f - alpha) * ema_vel_x_;
    ema_vel_y_ = alpha * kf_vy + (1.0f - alpha) * ema_vel_y_;
    mean_[4] = ema_vel_x_;
    mean_[5] = ema_vel_y_;
}
