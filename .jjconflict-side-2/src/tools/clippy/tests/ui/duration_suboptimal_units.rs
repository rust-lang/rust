//@aux-build:proc_macros.rs
#![warn(clippy::duration_suboptimal_units)]

use std::time::Duration;

const SIXTY: u64 = 60;

macro_rules! mac {
    (slow_rhythm) => {
        60 * 60
    };
    (duration) => {
        Duration::from_secs(60 * 5)
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

    let dur = Duration::from_secs(60 * 3);
    //~^ duration_suboptimal_units
    let dur = Duration::from_secs(10 * 60);
    //~^ duration_suboptimal_units
    let dur = Duration::from_mins(24 * 60);
    //~^ duration_suboptimal_units
    let dur = Duration::from_millis(5 * 1000);
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
        let dur = Duration::from_millis(1000 * 5);
        //~^ duration_suboptimal_units
        let dur = Duration::from_millis(11000);
        //~^ duration_suboptimal_units

        let dur = Duration::from_secs(180);
        // 39600 secs = 11 hours
        let dur = Duration::from_secs(39600);
        //~^ duration_suboptimal_units
        let dur = Duration::from_secs(3 * 60);
        //~^ duration_suboptimal_units
        let dur = Duration::from_mins(24 * 60);
        //~^ duration_suboptimal_units

        let dur = Duration::from_secs(SIXTY);
    }

    // Qualified Durations must be kept
    std::time::Duration::from_secs(12 * 60);
    //~^ duration_suboptimal_units

    // We lint in normal macros
    assert_eq!(Duration::from_secs(3_600), Duration::from_mins(6));
    assert_eq!(Duration::from_secs(60 * 60), Duration::from_mins(6));
    //~^ duration_suboptimal_units

    // We lint in normal macros (marker is in macro itself)
    let dur = mac!(duration);

    // We don't lint in macros if duration comes from outside
    let dur = mac!(arg => 60 * 60);

    // We don't lint in external macros
    let dur = proc_macros::external! { Duration::from_secs(60 * 60) };

    // We don't lint values coming from macros
    let dur = Duration::from_secs(mac!(slow_rhythm));
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

fn issue16457() {
    // Methods taking something else than `u64` are not covered
    _ = Duration::from_nanos_u128(1 << 90);
}

#[clippy::msrv = "1.90"]
fn insufficient_msrv() {
    _ = Duration::from_secs(67_768_040_922_076_800);
}

#[clippy::msrv = "1.91"]
fn sufficient_msrv() {
    _ = Duration::from_secs(67_768_040_922_076_800);
    //~^ duration_suboptimal_units
}

fn issue16532() {
    // Literals with small promoted values should not lint (issue #16532)
    let dur = Duration::from_secs(60);
    let dur = Duration::from_secs(180);
    let dur = Duration::from_millis(5_000);
    let dur = Duration::from_millis(1_000);
    let dur = Duration::from_secs(3_600);
    let dur = Duration::from_secs(600);

    // Literals with larger promoted values should lint
    let dur = Duration::from_millis(20_000);
    //~^ duration_suboptimal_units
    let dur = Duration::from_mins(720);
    //~^ duration_suboptimal_units
    let dur = Duration::from_secs(660);
    //~^ duration_suboptimal_units

    // Expressions should always be promoted, as they show intent to use a larger unit
    // 5 minutes
    let dur = Duration::from_secs(60 * 5);
    //~^ duration_suboptimal_units

    // 2 Hours
    let dur = Duration::from_mins(60 * 2);
    //~^ duration_suboptimal_units

    // Daily
    let dur = Duration::from_mins(60 * 24);
    //~^ duration_suboptimal_units
}
