import Only_whites_or_ten_fingers as img
import threading

from test import TargetTracker



def main():

    tracker = img.MasterTracker()

    while True:
        result = tracker.track()
        if result["target_locked"]:
            cx = result["center_x"]
            cy = result["center_y"]
            # Use cx, cy to compute yaw_rate and pitch
            print(f"Target at ({cx}, {cy})")
        else:
            print("No target locked.")


if __name__ == main():
    main()