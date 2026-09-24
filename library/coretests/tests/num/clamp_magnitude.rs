use core::num::*;

macro_rules! check_int_clamp {
    ($t:ty, $ut:ty) => {
        let min = <$t>::MIN;
        let max = <$t>::MAX;
        let max_u = <$ut>::MAX;

        // Basic clamping
        assert_eq!((100 as $t).clamp_magnitude(50), 50);
        assert_eq!((-100 as $t).clamp_magnitude(50), -50);
        assert_eq!((30 as $t).clamp_magnitude(50), 30);
        assert_eq!((-30 as $t).clamp_magnitude(50), -30);

        // Exact boundary
        assert_eq!((50 as $t).clamp_magnitude(50), 50);
        assert_eq!((-50 as $t).clamp_magnitude(50), -50);

        // Zero cases
        assert_eq!((0 as $t).clamp_magnitude(100), 0);
        assert_eq!((0 as $t).clamp_magnitude(0), 0);
        assert_eq!((100 as $t).clamp_magnitude(0), 0);
        assert_eq!((-100 as $t).clamp_magnitude(0), 0);

        // MIN/MAX values
        // Symmetric range [-MAX, MAX]
        assert_eq!(max.clamp_magnitude(max as $ut), max);
        assert_eq!(min.clamp_magnitude(max as $ut), -max);

        // Full range (limit covers MIN)
        let min_abs = min.unsigned_abs();
        assert_eq!(min.clamp_magnitude(min_abs), min);

        // Limit larger than type max (uN > iN::MAX)
        assert_eq!(max.clamp_magnitude(max_u), max);
        assert_eq!(min.clamp_magnitude(max_u), min);

        // Const clamping
        const C1: $t = (100 as $t).clamp_magnitude(50);
        assert_eq!(C1, 50);
        const C2: $t = (-100 as $t).clamp_magnitude(50);
        assert_eq!(C2, -50);
        const C3: $t = (30 as $t).clamp_magnitude(50);
        assert_eq!(C3, 30);
        const C4: $t = (-30 as $t).clamp_magnitude(50);
        assert_eq!(C4, -30);
    };
}

#[test]
fn test_clamp_magnitude_i8() {
    check_int_clamp!(i8, u8);
}

#[test]
fn test_clamp_magnitude_i16() {
    check_int_clamp!(i16, u16);
}

#[test]
fn test_clamp_magnitude_i32() {
    check_int_clamp!(i32, u32);
}

#[test]
fn test_clamp_magnitude_i64() {
    check_int_clamp!(i64, u64);
}

#[test]
fn test_clamp_magnitude_i128() {
    check_int_clamp!(i128, u128);
}

#[test]
fn test_clamp_magnitude_isize() {
    check_int_clamp!(isize, usize);
}

macro_rules! check_nonzero_clamp {
    ($t:ty, $ut:ty) => {
        let min = <$t>::MIN;
        let max = <$t>::MAX;
        let max_u = <$ut>::MAX;

        // Basic clamping
        let lim = <$ut>::new(50).unwrap();
        assert_eq!(<$t>::new(100).unwrap().clamp_magnitude(lim), <$t>::new(50).unwrap());
        assert_eq!(<$t>::new(-100).unwrap().clamp_magnitude(lim), <$t>::new(-50).unwrap());
        assert_eq!(<$t>::new(30).unwrap().clamp_magnitude(lim), <$t>::new(30).unwrap());
        assert_eq!(<$t>::new(-30).unwrap().clamp_magnitude(lim), <$t>::new(-30).unwrap());

        // Exact boundary
        assert_eq!(<$t>::new(50).unwrap().clamp_magnitude(lim), <$t>::new(50).unwrap());
        assert_eq!(<$t>::new(-50).unwrap().clamp_magnitude(lim), <$t>::new(-50).unwrap());

        // MIN/MAX values
        // Symmetric range [-MAX, MAX]
        assert_eq!(max.clamp_magnitude(max.try_into().unwrap()), max);
        assert_eq!(min.clamp_magnitude(max.try_into().unwrap()), -max);

        // Full range (limit covers MIN)
        let min_abs = min.unsigned_abs();
        assert_eq!(min.clamp_magnitude(min_abs), min);

        // Limit larger than type max (uN > iN::MAX)
        assert_eq!(max.clamp_magnitude(max_u), max);
        assert_eq!(min.clamp_magnitude(max_u), min);

        // Const clamping
        const LIM: $ut = <$ut>::new(50).unwrap();
        const C1: $t = <$t>::new(100).unwrap().clamp_magnitude(LIM);
        assert_eq!(C1, <$t>::new(50).unwrap());
        const C2: $t = <$t>::new(-100).unwrap().clamp_magnitude(LIM);
        assert_eq!(C2, <$t>::new(-50).unwrap());
        const C3: $t = <$t>::new(30).unwrap().clamp_magnitude(LIM);
        assert_eq!(C3, <$t>::new(30).unwrap());
        const C4: $t = <$t>::new(-30).unwrap().clamp_magnitude(LIM);
        assert_eq!(C4, <$t>::new(-30).unwrap());
    };
}

#[test]
fn test_clamp_magnitude_nonzero_i8() {
    check_nonzero_clamp!(NonZeroI8, NonZeroU8);
}

#[test]
fn test_clamp_magnitude_nonzero_i16() {
    check_nonzero_clamp!(NonZeroI16, NonZeroU16);
}

#[test]
fn test_clamp_magnitude_nonzero_i32() {
    check_nonzero_clamp!(NonZeroI32, NonZeroU32);
}

#[test]
fn test_clamp_magnitude_nonzero_i64() {
    check_nonzero_clamp!(NonZeroI64, NonZeroU64);
}

#[test]
fn test_clamp_magnitude_nonzero_i128() {
    check_nonzero_clamp!(NonZeroI128, NonZeroU128);
}

#[test]
fn test_clamp_magnitude_nonzero_isize() {
    check_nonzero_clamp!(NonZeroIsize, NonZeroUsize);
}

macro_rules! check_float_clamp {
    ($t:ty) => {
        // Basic clamping
        assert_eq!((5.0 as $t).clamp_magnitude(3.0), 3.0);
        assert_eq!((-5.0 as $t).clamp_magnitude(3.0), -3.0);
        assert_eq!((2.0 as $t).clamp_magnitude(3.0), 2.0);
        assert_eq!((-2.0 as $t).clamp_magnitude(3.0), -2.0);

        // Exact boundary
        assert_eq!((3.0 as $t).clamp_magnitude(3.0), 3.0);
        assert_eq!((-3.0 as $t).clamp_magnitude(3.0), -3.0);

        // Zero cases
        assert_eq!((0.0 as $t).clamp_magnitude(1.0), 0.0);
        assert_eq!((-0.0 as $t).clamp_magnitude(1.0), 0.0);
        assert_eq!((5.0 as $t).clamp_magnitude(0.0), 0.0);
        assert_eq!((-5.0 as $t).clamp_magnitude(0.0), 0.0);

        // Special values - Infinity
        let inf = <$t>::INFINITY;
        let neg_inf = <$t>::NEG_INFINITY;
        assert_eq!(inf.clamp_magnitude(100.0), 100.0);
        assert_eq!(neg_inf.clamp_magnitude(100.0), -100.0);
        assert_eq!(inf.clamp_magnitude(inf), inf);

        // Value with infinite limit
        assert_eq!((1.0 as $t).clamp_magnitude(inf), 1.0);
        assert_eq!((-1.0 as $t).clamp_magnitude(inf), -1.0);

        // MIN and MAX
        let max = <$t>::MAX;
        let min = <$t>::MIN;
        // Large limit
        let huge = 1e30;
        assert_eq!(max.clamp_magnitude(huge), huge);
        assert_eq!(min.clamp_magnitude(huge), -huge);

        // Const clamping
        const C1: $t = (5.0 as $t).clamp_magnitude(3.0);
        assert_eq!(C1, 3.0);
        const C2: $t = (-5.0 as $t).clamp_magnitude(3.0);
        assert_eq!(C2, -3.0);
        const C3: $t = (2.0 as $t).clamp_magnitude(3.0);
        assert_eq!(C3, 2.0);
        const C4: $t = (-2.0 as $t).clamp_magnitude(3.0);
        assert_eq!(C4, -2.0);
    };
}

#[test]
fn test_clamp_magnitude_f32() {
    check_float_clamp!(f32);
}

#[test]
fn test_clamp_magnitude_f64() {
    check_float_clamp!(f64);
}

#[test]
#[should_panic(expected = "limit must be non-negative and not NaN")]
fn test_clamp_magnitude_f32_panic_negative_limit() {
    let _ = 1.0f32.clamp_magnitude(-1.0);
}

#[test]
#[should_panic(expected = "limit must be non-negative and not NaN")]
fn test_clamp_magnitude_f64_panic_negative_limit() {
    let _ = 1.0f64.clamp_magnitude(-1.0);
}

#[test]
#[should_panic(expected = "limit must be non-negative and not NaN")]
fn test_clamp_magnitude_f32_panic_nan_limit() {
    let _ = 1.0f32.clamp_magnitude(f32::NAN);
}

#[test]
#[should_panic(expected = "limit must be non-negative and not NaN")]
fn test_clamp_magnitude_f64_panic_nan_limit() {
    let _ = 1.0f64.clamp_magnitude(f64::NAN);
}
