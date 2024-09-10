import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import euclidean
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s', filename='shot_rating.log')
logger = logging.getLogger(__name__)


def extract_data(frame_index, ball_track, homography_matrices, persons_top, persons_bottom, bounces, fps,
                 previous_shot=None):
    logger.info(f"\nExtracting data for frame {frame_index}")

    # Calculate ball speed
    ball_speed = 0
    if frame_index > 0:
        current_pos = ball_track[frame_index]
        previous_pos = ball_track[frame_index - 1]
        if current_pos is not None and previous_pos is not None and len(current_pos) == 2 and len(previous_pos) == 2:
            try:
                distance = euclidean(current_pos, previous_pos)
                ball_speed = distance * fps * 3.6  # Convert to km/h
            except Exception as e:
                logger.warning(f"Error calculating ball speed: {e}")
    logger.info(f"Ball speed: {ball_speed:.2f} km/h")

    # Estimate ball spin (unchanged)
    ball_spin = ball_speed * 20  # Rough estimate
    logger.info(f"Estimated ball spin: {ball_spin:.2f} RPM")

    # Calculate ball depth and width
    ball_depth = 0.5
    ball_width = 0.5
    if homography_matrices[frame_index] is not None and ball_track[frame_index] is not None:
        try:
            court_coords = np.array([[[ball_track[frame_index][0], ball_track[frame_index][1]]]], dtype=np.float32)
            transformed_coords = cv2.perspectiveTransform(court_coords, homography_matrices[frame_index])
            ball_depth = transformed_coords[0][0][1] / 1200  # Assuming court length is 2400 cm
            ball_width = transformed_coords[0][0][0] / 1098  # Assuming court width is 1098 cm (doubles court)
        except Exception as e:
            logger.warning(f"Error calculating ball depth and width: {e}")
    logger.info(f"Ball depth: {ball_depth:.2f}, Ball width: {ball_width:.2f}")

    # Determine player position (unchanged)
    if persons_bottom[frame_index] and len(persons_bottom[frame_index][0]) > 0:
        player_bbox = persons_bottom[frame_index][0][0]
        player_position = (player_bbox[1] + player_bbox[3]) / 2 / 1080  # Assuming 1080p video
    else:
        player_position = 0.5
    logger.info(f"Player position: {player_position:.2f}")

    # Determine shot type (unchanged)
    if frame_index in bounces:
        shot_type = "Groundstroke"
    elif previous_shot and previous_shot['type'] == "Serve":
        shot_type = "Return"
    elif not previous_shot:
        shot_type = "Serve"
    else:
        shot_type = "Groundstroke"

    # Further classify groundstrokes
    if shot_type == "Groundstroke":
        shot_type = "Forehand" if np.random.random() > 0.5 else "Backhand"
    logger.info(f"Determined shot type: {shot_type}")

    return ball_speed, ball_spin, ball_depth, ball_width, player_position, shot_type, previous_shot


def calculate_shot_quality(ball_speed, ball_spin, ball_depth, ball_width, shot_type, previous_shot):
    logger.info("\nCalculating shot quality")

    ideal_ranges = {
        'Serve': {'speed': (180, 30), 'spin': (3000, 1000), 'depth': (0.9, 0.1), 'width': (0.9, 0.1)},
        'Return': {'speed': (120, 30), 'spin': (2500, 1000), 'depth': (0.8, 0.1), 'width': (0.8, 0.1)},
        'Forehand': {'speed': (150, 30), 'spin': (3500, 1000), 'depth': (0.85, 0.1), 'width': (0.85, 0.1)},
        'Backhand': {'speed': (140, 30), 'spin': (3000, 1000), 'depth': (0.8, 0.1), 'width': (0.8, 0.1)}
    }

    # Calculate individual quality scores
    speed_score = norm.pdf(ball_speed, *ideal_ranges[shot_type]['speed'])
    spin_score = norm.pdf(ball_spin, *ideal_ranges[shot_type]['spin'])
    depth_score = norm.pdf(ball_depth, *ideal_ranges[shot_type]['depth'])
    width_score = norm.pdf(ball_width, *ideal_ranges[shot_type]['width'])

    logger.info(f"Raw scores - Speed: {speed_score:.4f}, Spin: {spin_score:.4f}, Depth: {depth_score:.4f}, Width: {width_score:.4f}")

    # Normalize scores to 0-10 range
    speed_score = (speed_score / norm.pdf(ideal_ranges[shot_type]['speed'][0], *ideal_ranges[shot_type]['speed'])) * 10
    spin_score = (spin_score / norm.pdf(ideal_ranges[shot_type]['spin'][0], *ideal_ranges[shot_type]['spin'])) * 10
    depth_score = (depth_score / norm.pdf(ideal_ranges[shot_type]['depth'][0], *ideal_ranges[shot_type]['depth'])) * 10
    width_score = (width_score / norm.pdf(ideal_ranges[shot_type]['width'][0], *ideal_ranges[shot_type]['width'])) * 10

    logger.info(f"Normalized scores - Speed: {speed_score:.2f}, Spin: {spin_score:.2f}, Depth: {depth_score:.2f}, Width: {width_score:.2f}")

    # Calculate overall shot quality with weighted average
    weights = {'speed': 0.3, 'spin': 0.2, 'depth': 0.3, 'width': 0.2}
    shot_quality = (
        speed_score * weights['speed'] +
        spin_score * weights['spin'] +
        depth_score * weights['depth'] +
        width_score * weights['width']
    )
    logger.info(f"Initial shot quality: {shot_quality:.2f}")

    # Adjust for previous shot quality
    if previous_shot:
        difficulty_adjustment = min(1.5, max(0.5, 1 + (previous_shot['quality'] - 5) / 10))
        shot_quality *= difficulty_adjustment
        logger.info(f"Adjusted for previous shot. Difficulty adjustment: {difficulty_adjustment:.2f}")

    shot_quality = min(10, max(0, shot_quality))
    logger.info(f"Final shot quality: {shot_quality:.2f}")

    return shot_quality


def classify_shot_type(shot_quality, player_position, ball_speed, ball_spin, ball_depth, ball_width):
    logger.info("\nClassifying shot type")
    logger.info(f"Inputs - Quality: {shot_quality:.2f}, Player Position: {player_position:.2f}, "
                f"Ball Speed: {ball_speed:.2f}, Ball Spin: {ball_spin:.2f}, Ball Depth: {ball_depth:.2f}, Ball Width: {ball_width:.2f}")

    # Define thresholds
    speed_threshold = 140
    spin_threshold = 3500
    depth_threshold = 0.8
    width_threshold = 0.2  # Distance from sideline

    if shot_quality < 3:
        if ball_depth < 0.5 or ball_width < 0.1 or ball_width > 0.9:
            classification = "Unforced Error"
        else:
            classification = "Weak Shot"
    elif shot_quality > 8:
        if ball_speed > speed_threshold and ball_depth > depth_threshold:
            classification = "Winner"
        elif ball_spin > spin_threshold and ball_depth > depth_threshold:
            classification = "Heavy Topspin"
        elif abs(ball_width - 0.5) < width_threshold:
            classification = "Down the Line"
        else:
            classification = "Aggressive Shot"
    else:
        if player_position < 0.3:
            classification = "Defensive Shot"
        elif player_position > 0.7:
            classification = "Approach Shot"
        elif ball_depth > depth_threshold and ball_speed < speed_threshold - 20:
            classification = "Deep Rally Ball"
        elif ball_depth < 0.3 and ball_speed < speed_threshold - 40:
            classification = "Drop Shot"
        else:
            classification = "Neutral Rally"

    logger.info(f"Shot classified as: {classification}")
    return classification


def analyze_shot(frame_index, ball_track, homography_matrices, persons_top, persons_bottom, bounces, fps,
                 previous_shot=None):
    logger.info(f"\n{'=' * 50}\nAnalyzing shot for frame {frame_index}\n{'=' * 50}")

    ball_speed, ball_spin, ball_depth, ball_width, player_position, shot_type, _ = extract_data(
        frame_index, ball_track, homography_matrices, persons_top, persons_bottom, bounces, fps, previous_shot
    )

    shot_quality = calculate_shot_quality(ball_speed, ball_spin, ball_depth, ball_width, shot_type, previous_shot)
    shot_classification = classify_shot_type(shot_quality, player_position, ball_speed, ball_spin, ball_depth)

    logger.info(f"\nFinal shot analysis:")
    logger.info(f"Shot Quality: {shot_quality:.2f}")
    logger.info(f"Shot Type: {shot_type}")
    logger.info(f"Classification: {shot_classification}")
    logger.info(f"Ball Speed: {ball_speed:.2f} km/h")
    logger.info(f"Ball Spin: {ball_spin:.2f} RPM")
    logger.info(f"Ball Depth: {ball_depth:.2f}")
    logger.info(f"Ball Width: {ball_width:.2f}")
    logger.info(f"Player Position: {player_position:.2f}")

    return {
        'quality': shot_quality,
        'type': shot_type,
        'classification': shot_classification,
        'speed': ball_speed,
        'spin': ball_spin,
        'depth': ball_depth,
        'width': ball_width,
        'player_position': player_position
    }


# Example usage
if __name__ == "__main__":
    # This is a mock example. In real usage, you would pass actual tracking data.
    mock_ball_track = [((i * 10) % 100, (i * 10) % 100) for i in range(100)]
    mock_homography_matrices = [np.eye(3) for _ in range(100)]
    mock_persons_top = [[[(0, 0, 100, 100)]]] * 100
    mock_persons_bottom = [[[(0, 980, 100, 1080)]]] * 100
    mock_bounces = [20, 40, 60, 80]
    mock_fps = 30

    previous_shot = None
    for i in range(5):  # Simulate 5 shots in a rally
        shot_data = analyze_shot(i * 10, mock_ball_track, mock_homography_matrices,
                                 mock_persons_top, mock_persons_bottom, mock_bounces,
                                 mock_fps, previous_shot)
        previous_shot = shot_data
        print("\n" + "=" * 50)  # Add a separator between shots