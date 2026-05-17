#pragma once

#include "ByteTrack/Rect.h"
#include "ByteTrack/KalmanFilter.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace byte_track
{
enum class STrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3,
};

class STrack
{
public:
    STrack(const Rect<float>& rect, const float& score);
    ~STrack();

    const Rect<float>& getRect() const;
    const STrackState& getSTrackState() const;

    const bool& isActivated() const;
    const float& getScore() const;
    const size_t& getTrackId() const;
    const size_t& getFrameId() const;
    const size_t& getStartFrameId() const;
    const size_t& getTrackletLength() const;

    int64_t getLastSeenTimestampMs() const { return last_seen_timestamp_ms_; }
    void setLastSeenTimestampMs(int64_t timestamp_ms) { last_seen_timestamp_ms_ = timestamp_ms; }

    void activate(const size_t& frame_id, const size_t& track_id);
    void reActivate(const STrack &new_track, const size_t &frame_id,
                    const int &new_track_id = -1, float vel_ema_alpha = 1.0f);

    void predict(float dt = 1.0f, float tracked_dt_cap = 3.0f, float lost_dt_cap = 2.0f);
    void update(const STrack &new_track, const size_t &frame_id, float vel_ema_alpha = 1.0f);

    std::pair<float, float> getVelocity() const;

    void markAsLost();
    void markAsRemoved();

private:
    KalmanFilter kalman_filter_;
    KalmanFilter::StateMean mean_;
    KalmanFilter::StateCov covariance_;

    Rect<float> rect_;
    STrackState state_;

    bool is_activated_;
    float score_;
    size_t track_id_;
    size_t frame_id_;
    size_t start_frame_id_;
    size_t tracklet_len_;
    int64_t last_seen_timestamp_ms_{-1};

    float ema_vel_x_{0.0f};
    float ema_vel_y_{0.0f};
    bool ema_initialized_{false};

    void updateRect();
    void applyVelocityEma(float alpha);
};
}