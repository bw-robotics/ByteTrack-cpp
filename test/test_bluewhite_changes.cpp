#include "ByteTrack/BYTETracker.h"
#include "ByteTrack/KalmanFilter.h"
#include "ByteTrack/STrack.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace
{

byte_track::Object makeObject(float x, float y, float w, float h, float prob)
{
    return byte_track::Object(byte_track::Rect<float>(x, y, w, h), 0, prob);
}

std::vector<byte_track::Object> makeSingleDetection(float x, float y, float w, float h, float prob)
{
    return {makeObject(x, y, w, h, prob)};
}

}  // namespace

// --- ByteTrackerConfig tests ---

TEST(ByteTrackerConfig, DefaultValues)
{
    byte_track::ByteTrackerConfig config;
    EXPECT_EQ(config.frame_rate, 30);
    EXPECT_EQ(config.track_buffer, 30);
    EXPECT_FLOAT_EQ(config.track_thresh, 0.5f);
    EXPECT_FLOAT_EQ(config.high_thresh, 0.6f);
    EXPECT_FLOAT_EQ(config.match_thresh, 0.8f);
    EXPECT_FLOAT_EQ(config.max_area_ratio, 3.0f);
    EXPECT_FLOAT_EQ(config.low_score_match_thresh, 0.5f);
    EXPECT_FLOAT_EQ(config.unconfirmed_match_thresh, 0.7f);
    EXPECT_FLOAT_EQ(config.tracked_predict_dt_cap, 3.0f);
    EXPECT_FLOAT_EQ(config.lost_predict_dt_cap, 2.0f);
}

TEST(ByteTrackerConfig, AsString)
{
    byte_track::ByteTrackerConfig config;
    std::string s = config.asString();
    EXPECT_NE(s.find("frame_rate="), std::string::npos);
    EXPECT_NE(s.find("track_buffer="), std::string::npos);
    EXPECT_NE(s.find("max_area_ratio="), std::string::npos);
    EXPECT_NE(s.find("tracked_predict_dt_cap="), std::string::npos);
    EXPECT_NE(s.find("lost_predict_dt_cap="), std::string::npos);
}

TEST(ByteTrackerConfig, CustomValues)
{
    byte_track::ByteTrackerConfig config;
    config.frame_rate = 60;
    config.track_buffer = 60;
    config.track_thresh = 0.3f;

    byte_track::BYTETracker tracker(config);
    EXPECT_FLOAT_EQ(tracker.getExpectedDtMs(), 1000.0f / 60.0f);
}

// --- BYTETracker construction and basic API ---

TEST(BYTETracker, DefaultConstruction)
{
    byte_track::BYTETracker tracker;
    EXPECT_FLOAT_EQ(tracker.getCurrentDt(), 1.0f);
    EXPECT_FLOAT_EQ(tracker.getExpectedDtMs(), 1000.0f / 30.0f);
    EXPECT_TRUE(tracker.getLostTracks().empty());
    EXPECT_TRUE(tracker.getAllActiveTracks().empty());
}

TEST(BYTETracker, EmptyUpdate)
{
    byte_track::BYTETracker tracker;
    auto result = tracker.update({});
    EXPECT_TRUE(result.empty());
}

TEST(BYTETracker, SingleDetectionCreatesTrack)
{
    byte_track::ByteTrackerConfig config;
    config.high_thresh = 0.5f;
    byte_track::BYTETracker tracker(config);

    auto dets = makeSingleDetection(100, 100, 50, 80, 0.9f);

    // Frame 1: detection creates unconfirmed track (not yet activated for output)
    auto result1 = tracker.update(dets);
    // Frame 2: same detection confirms and activates track
    auto result2 = tracker.update(dets);
    EXPECT_EQ(result2.size(), 1u);
    EXPECT_EQ(result2[0]->getTrackId(), 1u);
}

// --- Variable dt via timestamps ---

TEST(BYTETracker, TimestampDtComputation)
{
    byte_track::ByteTrackerConfig config;
    config.frame_rate = 30;
    byte_track::BYTETracker tracker(config);

    auto dets = makeSingleDetection(100, 100, 50, 80, 0.9f);

    // First call: no dt yet (default 1.0)
    tracker.update(dets, 0);
    EXPECT_FLOAT_EQ(tracker.getCurrentDt(), 1.0f);

    // Second call at exactly expected interval: dt should be 1.0
    int64_t expected_ms = static_cast<int64_t>(1000.0f / 30.0f);
    tracker.update(dets, expected_ms);
    EXPECT_NEAR(tracker.getCurrentDt(), 1.0f, 0.01f);

    // Third call at double interval: dt should be ~2.0
    tracker.update(dets, expected_ms + 2 * expected_ms);
    EXPECT_NEAR(tracker.getCurrentDt(), 2.0f, 0.1f);
}

TEST(BYTETracker, NoTimestampDefaultsDtToOne)
{
    byte_track::BYTETracker tracker;
    auto dets = makeSingleDetection(100, 100, 50, 80, 0.9f);

    tracker.update(dets);
    EXPECT_FLOAT_EQ(tracker.getCurrentDt(), 1.0f);
    tracker.update(dets);
    EXPECT_FLOAT_EQ(tracker.getCurrentDt(), 1.0f);
}

// --- KalmanFilter variable dt ---

TEST(KalmanFilter, PredictDtOne)
{
    byte_track::KalmanFilter kf;
    byte_track::KalmanFilter::StateMean mean;
    byte_track::KalmanFilter::StateCov cov;

    byte_track::Xyah<float> measurement;
    measurement << 100.0f, 200.0f, 0.5f, 300.0f;
    kf.initiate(mean, cov, measurement);

    byte_track::KalmanFilter::StateMean mean_dt1 = mean;
    byte_track::KalmanFilter::StateCov cov_dt1 = cov;
    kf.predict(mean_dt1, cov_dt1, 1.0f);

    byte_track::KalmanFilter::StateMean mean_default = mean;
    byte_track::KalmanFilter::StateCov cov_default = cov;
    kf.predict(mean_default, cov_default);

    for (int i = 0; i < 8; i++)
    {
        EXPECT_NEAR(mean_dt1(i), mean_default(i), 1e-5f) << "Mismatch at index " << i;
    }
}

TEST(KalmanFilter, PredictLargerDtMovesMore)
{
    byte_track::KalmanFilter kf;
    byte_track::KalmanFilter::StateMean mean;
    byte_track::KalmanFilter::StateCov cov;

    byte_track::Xyah<float> measurement;
    measurement << 100.0f, 200.0f, 0.5f, 300.0f;
    kf.initiate(mean, cov, measurement);

    // Set a velocity in mean
    mean(4) = 10.0f;  // vx
    mean(5) = 5.0f;   // vy

    byte_track::KalmanFilter::StateMean mean_dt1 = mean;
    byte_track::KalmanFilter::StateCov cov_dt1 = cov;
    kf.predict(mean_dt1, cov_dt1, 1.0f);

    byte_track::KalmanFilter::StateMean mean_dt2 = mean;
    byte_track::KalmanFilter::StateCov cov_dt2 = cov;
    kf.predict(mean_dt2, cov_dt2, 2.0f);

    // With dt=2 the position should shift more
    float dx_dt1 = mean_dt1(0) - mean(0);
    float dx_dt2 = mean_dt2(0) - mean(0);
    EXPECT_GT(std::abs(dx_dt2), std::abs(dx_dt1));

    // Covariance should also be larger with bigger dt
    EXPECT_GT(cov_dt2(0, 0), cov_dt1(0, 0));
}

// --- STrack predict with dt ---

TEST(STrack, PredictDefaultDt)
{
    byte_track::Rect<float> rect(100.0f, 200.0f, 50.0f, 80.0f);
    byte_track::STrack track1(rect, 0.9f);
    byte_track::STrack track2(rect, 0.9f);

    track1.activate(1, 1);
    track2.activate(1, 2);

    track1.predict();
    track2.predict(1.0f);

    EXPECT_NEAR(track1.getRect().x(), track2.getRect().x(), 1e-4f);
    EXPECT_NEAR(track1.getRect().y(), track2.getRect().y(), 1e-4f);
}

TEST(STrack, GetVelocity)
{
    byte_track::Rect<float> rect(100.0f, 200.0f, 50.0f, 80.0f);
    byte_track::STrack track(rect, 0.9f);
    track.activate(1, 1);

    auto [vx, vy] = track.getVelocity();
    // After initialization velocities are zero
    EXPECT_NEAR(vx, 0.0f, 1e-5f);
    EXPECT_NEAR(vy, 0.0f, 1e-5f);
}

// --- getLostTracks / getAllActiveTracks ---

TEST(BYTETracker, LostTracksAfterDisappearance)
{
    byte_track::ByteTrackerConfig config;
    config.high_thresh = 0.5f;
    config.track_thresh = 0.4f;
    byte_track::BYTETracker tracker(config);

    auto dets = makeSingleDetection(100, 100, 50, 80, 0.9f);

    // Build up a confirmed track over several frames
    for (int i = 0; i < 5; i++)
    {
        tracker.update(dets);
    }
    EXPECT_EQ(tracker.getLostTracks().size(), 0u);

    // Object disappears - track should become lost
    tracker.update({});
    EXPECT_GE(tracker.getLostTracks().size(), 1u);
}

TEST(BYTETracker, GetAllActiveTracksIncludesLost)
{
    byte_track::ByteTrackerConfig config;
    config.high_thresh = 0.5f;
    config.track_thresh = 0.4f;
    byte_track::BYTETracker tracker(config);

    auto dets = makeSingleDetection(100, 100, 50, 80, 0.9f);

    for (int i = 0; i < 5; i++)
    {
        tracker.update(dets);
    }
    size_t active_before = tracker.getAllActiveTracks().size();
    EXPECT_GE(active_before, 1u);

    // Object disappears
    tracker.update({});
    size_t active_after = tracker.getAllActiveTracks().size();
    // getAllActiveTracks should include both tracked and lost
    EXPECT_GE(active_after, 1u);
}

// --- Area ratio constraint ---

TEST(BYTETracker, AreaRatioRejectsLargeSizeDifference)
{
    byte_track::ByteTrackerConfig config;
    config.max_area_ratio = 2.0f;
    config.high_thresh = 0.5f;
    config.track_thresh = 0.4f;
    byte_track::BYTETracker tracker(config);

    auto small_det = makeSingleDetection(100, 100, 30, 30, 0.9f);
    // Establish track
    for (int i = 0; i < 3; i++)
    {
        tracker.update(small_det);
    }
    auto tracked_before = tracker.update(small_det);
    ASSERT_GE(tracked_before.size(), 1u);
    size_t id_before = tracked_before[0]->getTrackId();

    // Now present a much larger detection in similar position
    auto big_det = makeSingleDetection(100, 100, 200, 200, 0.9f);
    auto tracked_after = tracker.update(big_det);

    // The large detection should not match the small track due to area ratio
    // Instead it should create a new track (different id) or not match
    bool same_id_matched = false;
    for (const auto &t : tracked_after)
    {
        if (t->getTrackId() == id_before)
        {
            // If matched, the rect should be close to the small one (predicted), not the big one
            float area = t->getRect().width() * t->getRect().height();
            if (area > 30 * 30 * 4.0f)
            {
                same_id_matched = true;
            }
        }
    }
    EXPECT_FALSE(same_id_matched) << "Large detection should not match small track with area_ratio=2.0";
}

// --- Configurable thresholds ---

TEST(BYTETracker, ConfigurableTrackThresh)
{
    // With high track_thresh, low-confidence detections should be ignored
    byte_track::ByteTrackerConfig config;
    config.track_thresh = 0.8f;
    config.high_thresh = 0.8f;
    byte_track::BYTETracker tracker(config);

    auto low_conf_dets = makeSingleDetection(100, 100, 50, 80, 0.6f);
    for (int i = 0; i < 5; i++)
    {
        tracker.update(low_conf_dets);
    }

    auto result = tracker.update(low_conf_dets);
    EXPECT_EQ(result.size(), 0u) << "Detection below track_thresh should not create tracks";
}

TEST(BYTETracker, ConfigurableHighThresh)
{
    // With lower high_thresh, more detections become new tracks
    byte_track::ByteTrackerConfig config;
    config.track_thresh = 0.3f;
    config.high_thresh = 0.4f;
    byte_track::BYTETracker tracker(config);

    auto dets = makeSingleDetection(100, 100, 50, 80, 0.5f);

    for (int i = 0; i < 5; i++)
    {
        tracker.update(dets);
    }

    auto result = tracker.update(dets);
    EXPECT_GE(result.size(), 1u);
}

// --- max_time_lost (track_buffer) ---

TEST(BYTETracker, TrackBufferLimitsLostDuration)
{
    byte_track::ByteTrackerConfig config;
    config.track_buffer = 3;
    config.high_thresh = 0.5f;
    config.track_thresh = 0.4f;
    byte_track::BYTETracker tracker(config);

    auto dets = makeSingleDetection(100, 100, 50, 80, 0.9f);

    // Build confirmed track
    for (int i = 0; i < 5; i++)
    {
        tracker.update(dets);
    }

    // Object disappears; track should be lost for at most track_buffer frames
    for (int i = 0; i < 5; i++)
    {
        tracker.update({});
    }

    // After 5 empty frames with buffer=3, the lost track should be removed
    EXPECT_EQ(tracker.getLostTracks().size(), 0u);
}

// --- Multi-object tracking ---

TEST(BYTETracker, MultipleObjectsTracked)
{
    byte_track::ByteTrackerConfig config;
    config.high_thresh = 0.5f;
    config.track_thresh = 0.4f;
    byte_track::BYTETracker tracker(config);

    std::vector<byte_track::Object> dets = {
        makeObject(100, 100, 50, 80, 0.9f),
        makeObject(300, 300, 60, 90, 0.85f),
    };

    for (int i = 0; i < 5; i++)
    {
        tracker.update(dets);
    }

    auto result = tracker.update(dets);
    EXPECT_EQ(result.size(), 2u);

    // Track IDs should be different
    EXPECT_NE(result[0]->getTrackId(), result[1]->getTrackId());
}

// --- Backward compatibility: update without timestamp ---

TEST(BYTETracker, UpdateWithoutTimestampWorks)
{
    byte_track::BYTETracker tracker;
    auto dets = makeSingleDetection(100, 100, 50, 80, 0.9f);

    auto r1 = tracker.update(dets);
    auto r2 = tracker.update(dets);
    EXPECT_FLOAT_EQ(tracker.getCurrentDt(), 1.0f);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return (RUN_ALL_TESTS());
}
