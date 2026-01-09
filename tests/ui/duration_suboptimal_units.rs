//@aux-build:proc_macros.rs
#![warn(clippy::duration_suboptimal_units)]

use std::time::Duration;

const SIXTY: u64 = 60;

macro_rules! mac {
    (slow_rythm) => {
        3600
    };
    (duration) => {
        Duration::from_secs(300)
        //~^ duration_suboptimal_units
    };
    (arg => $e:expr) => {
        Duration::from_secs($e)
    };
}

fn main() {
    let dur = Duration::from_secs(0);
    let dur = Duration::from_secs(42);
    let dur = Duration::from_hours(3);

    let dur = Duration::from_secs(60);
    //~^ duration_suboptimal_units
    let dur = Duration::from_secs(180);
    //~^ duration_suboptimal_units
    let dur = Duration::from_secs(10 * 60);
    //~^ duration_suboptimal_units
    let dur = Duration::from_mins(24 * 60);
    //~^ duration_suboptimal_units
    let dur = Duration::from_millis(5_000);
    //~^ duration_suboptimal_units
    let dur = Duration::from_nanos(13 * 60 * 60 * 1_000 * 1_000 * 1_000);
    //~^ duration_suboptimal_units

    // Constants are intentionally not resolved, as we don't want to recommend a literal value over
    // using constants.
    let dur = Duration::from_secs(SIXTY);
    // Technically it would be nice to use Duration::from_mins(SIXTY) here, but that is a follow-up
    let dur = Duration::from_secs(SIXTY * 60);

    const {
        let dur = Duration::from_secs(0);
        let dur = Duration::from_millis(5_000);
        //~^ duration_suboptimal_units
        let dur = Duration::from_secs(180);
        //~^ duration_suboptimal_units
        let dur = Duration::from_mins(24 * 60);
        //~^ duration_suboptimal_units

        let dur = Duration::from_secs(SIXTY);
    }

    // Qualified Durations must be kept
    std::time::Duration::from_secs(60);
    //~^ duration_suboptimal_units

    // We lint in normal macros
    assert_eq!(Duration::from_secs(3_600), Duration::from_mins(6));
    //~^ duration_suboptimal_units

    // We lint in normal macros (marker is in macro itself)
    let dur = mac!(duration);

    // We don't lint in macros if duration comes from outside
    let dur = mac!(arg => 3600);

    // We don't lint in external macros
    let dur = proc_macros::external! { Duration::from_secs(3_600) };

    // We don't lint values coming from macros
    let dur = Duration::from_secs(mac!(slow_rythm));
}

mod my_duration {
    struct Duration {}

    impl Duration {
        pub const fn from_secs(_secs: u64) -> Self {
            Self {}
        }
    }

    fn test() {
        // Only suggest the change for std::time::Duration, not for other Duration structs
        let dur = Duration::from_secs(60);
    }
}
