use super::{can_monotonize_u64x2, monotonize_u64, monotonize_u64x2, MONOTONIZE_U64};
use crate::time::Instant;

// if the monotonizing functions are in use by the platform's Instant implementation
// the tests are more cautious and only check for inequalities or probe very small
// increments of time to not confuse other threads
// if they are not in use then the tests assume exclusive control over the static values

#[test]
fn monotonic_u64() {
    if !MONOTONIZE_U64 {
        assert_eq!(None, monotonize_u64(0));
        return;
    }

    let _ = Instant::now();

    let baseline = monotonize_u64(0).unwrap();

    if baseline == 0 {
        // monotonize_u64 does not appear to be in use by this platform's Instant impl
        assert_eq!(1, monotonize_u64(1).unwrap());
        assert_eq!(1, monotonize_u64(0).unwrap());
        assert_eq!(2, monotonize_u64(2).unwrap());
    } else {
        assert!(monotonize_u64(baseline - 1).unwrap() >= baseline);
        // this may steal the smallest representable amount of time from other threads
        assert!(monotonize_u64(baseline + 1).unwrap() >= baseline + 1);
    }
}

#[test]
fn monotonic_u64x2() {
    if !can_monotonize_u64x2() {
        assert_eq!(None, monotonize_u64x2(0, 0));
        return;
    }

    let _ = Instant::now();

    let baseline = monotonize_u64x2(0, 1).unwrap();

    if baseline == (0, 1) {
        // monotonize_u64x2 does not appear to be in use by this platform's Instant impl
        assert_eq!((1, 1), monotonize_u64x2(1, 1).unwrap());
        assert_eq!((1, 1), monotonize_u64x2(1, 0).unwrap());
        assert_eq!((1, 1), monotonize_u64x2(0, 2).unwrap());
        assert_eq!((2, 0), monotonize_u64x2(2, 0).unwrap());
    } else {
        let val = monotonize_u64x2(0, 0).unwrap();
        assert!(
            val.0 > baseline.0 || (val.0 == baseline.0 && val.1 >= baseline.0),
            "did not decrement"
        );
        let val = monotonize_u64x2(baseline.0 - 1, baseline.1 + 1).unwrap();
        assert!(
            val.0 > baseline.0 || (val.0 == baseline.0 && val.1 >= baseline.0),
            "there is no endianness confusion"
        );
    }
}
