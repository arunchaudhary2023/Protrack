import cv2
import numpy as np
import pandas as pd
import datetime
import os


def save_violation_log(violation_log_df, filename="violation_log.csv"):
    """
    Save the violation log dataframe to a CSV file in the logs directory.

    Args:
        violation_log_df: Pandas DataFrame containing violation log data
        filename: Name of the CSV file
    """
    # Ensure logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Save to CSV
    log_path = os.path.join('logs', filename)
    violation_log_df.to_csv(log_path, index=False)
    return log_path


def generate_sample_data(days=30):
    """
    Generate sample historical data for testing the dashboard.

    Args:
        days: Number of days of historical data to generate

    Returns:
        Tuple of DataFrames (violations, analytics, progress)
    """
    # Generate dates
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    dates = [start_date + datetime.timedelta(hours=i * 8) for i in range(days * 3)]

    # Generate violation data
    violation_types = ['Missing Helmet', 'Missing Vest', 'Unauthorized Access']
    violations = []

    for date in dates:
        if np.random.random() < 0.3:  # 30% chance of violation
            for _ in range(np.random.randint(0, 3)):
                violations.append({
                    'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
                    'type': np.random.choice(violation_types),
                    'description': f'Violation detected at {date.strftime("%H:%M:%S")}'
                })

    violations_df = pd.DataFrame(violations)

    # Generate analytics data
    analytics = []

    for date in dates:
        worker_count = np.random.randint(3, 15)
        helmet_pct = np.random.uniform(70, 100)
        vest_pct = np.random.uniform(75, 100)

        analytics.append({
            'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
            'worker_count': worker_count,
            'helmet_percentage': helmet_pct,
            'vest_percentage': vest_pct,
            'violation_count': np.random.randint(0, 3)
        })

    analytics_df = pd.DataFrame(analytics)

    # Generate progress data
    tasks = ['Foundation Work', 'Framing', 'Electrical', 'Plumbing']
    progress = []

    for date in dates[::3]:  # Less frequent progress updates
        for task in tasks:
            # Calculate days from start
            days_from_start = (date - start_date).days
            expected = min(100, (days_from_start / days) * 100)

            # Add some randomness to actual progress
            actual = max(0, min(100, expected + np.random.uniform(-15, 10)))

            progress.append({
                'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
                'task': task,
                'progress': actual,
                'expected_progress': expected
            })

    progress_df = pd.DataFrame(progress)

    return violations_df, analytics_df, progress_df


def convert_video_to_frames(video_path, output_dir="frames", fps=1):
    """
    Extract frames from a video file at the specified framerate.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract

    Returns:
        Number of frames extracted
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval
    frame_interval = int(video_fps / fps)

    # Extract frames
    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Extract frame at the specified interval
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            count += 1

        frame_count += 1

    cap.release()
    return count


def overlay_text_on_frame(frame, text, position=(10, 30), font_scale=0.7, color=(255, 255, 255), thickness=2):
    """
    Overlay text on a frame with a semi-transparent background for better readability.

    Args:
        frame: Image frame
        text: Text to overlay
        position: Position tuple (x, y)
        font_scale: Font scale
        color: Text color tuple (B, G, R)
        thickness: Text thickness

    Returns:
        Frame with overlaid text
    """
    # Make a copy of the frame
    result = frame.copy()

    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate background rectangle coordinates
    x, y = position
    bg_rect = [x - 5, y - text_height - 5, x + text_width + 5, y + 5]

    # Draw semi-transparent background
    overlay = result.copy()
    cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), (bg_rect[2], bg_rect[3]), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

    # Draw text
    cv2.putText(result, text, position, font, font_scale, color, thickness)

    return result


def add_timestamp(frame):
    """
    Add a timestamp to the frame.

    Args:
        frame: Image frame

    Returns:
        Frame with timestamp
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return overlay_text_on_frame(frame, timestamp, position=(10, frame.shape[0] - 10))