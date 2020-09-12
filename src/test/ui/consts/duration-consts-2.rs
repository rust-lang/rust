// run-pass

#![feature(const_panic)]
#![feature(duration_consts_2)]
#![feature(div_duration)]
#![feature(duration_saturating_ops)]

use std::time::Duration;

fn duration() {
    const ZERO : Duration = Duration::new(0, 0);
    assert_eq!(ZERO, Duration::from_secs(0));

    const ONE : Duration = Duration::new(0, 1);
    assert_eq!(ONE, Duration::from_nanos(1));

    const MAX : Duration = Duration::new(u64::MAX, 1_000_000_000 - 1);

    const MAX_CHECKED_ADD_ZERO : Option<Duration> = MAX.checked_add(ZERO);
    assert_eq!(MAX_CHECKED_ADD_ZERO, Some(MAX));

    const MAX_CHECKED_ADD_ONE : Option<Duration> = MAX.checked_add(ONE);
    assert_eq!(MAX_CHECKED_ADD_ONE, None);

    const ONE_CHECKED_SUB_ONE : Option<Duration> = ONE.checked_sub(ONE);
    assert_eq!(ONE_CHECKED_SUB_ONE, Some(ZERO));

    const ZERO_CHECKED_SUB_ONE : Option<Duration> = ZERO.checked_sub(ONE);
    assert_eq!(ZERO_CHECKED_SUB_ONE, None);

    const ONE_CHECKED_MUL_ONE : Option<Duration> = ONE.checked_mul(1);
    assert_eq!(ONE_CHECKED_MUL_ONE, Some(ONE));

    const MAX_CHECKED_MUL_TWO : Option<Duration> = MAX.checked_mul(2);
    assert_eq!(MAX_CHECKED_MUL_TWO, None);

    const ONE_CHECKED_DIV_ONE : Option<Duration> = ONE.checked_div(1);
    assert_eq!(ONE_CHECKED_DIV_ONE, Some(ONE));

    const ONE_CHECKED_DIV_ZERO : Option<Duration> = ONE.checked_div(0);
    assert_eq!(ONE_CHECKED_DIV_ZERO, None);

    const MAX_AS_F32 : f32 = MAX.as_secs_f32();
    assert_eq!(MAX_AS_F32, 18446744000000000000.0_f32);

    const MAX_AS_F64 : f64 = MAX.as_secs_f64();
    assert_eq!(MAX_AS_F64, 18446744073709552000.0_f64);

    const ONE_AS_F32 : f32 = ONE.div_duration_f32(ONE);
    assert_eq!(ONE_AS_F32, 1.0_f32);

    const ONE_AS_F64 : f64 = ONE.div_duration_f64(ONE);
    assert_eq!(ONE_AS_F64, 1.0_f64);

    const MAX_SATURATING_ADD_ONE : Duration = MAX.saturating_add(ONE);
    assert_eq!(MAX_SATURATING_ADD_ONE, MAX);

    const ZERO_SATURATING_SUB_ONE : Duration = ZERO.saturating_sub(ONE);
    assert_eq!(ZERO_SATURATING_SUB_ONE, ZERO);

    const MAX_SATURATING_MUL_TWO : Duration = MAX.saturating_mul(2);
    assert_eq!(MAX_SATURATING_MUL_TWO, MAX);
}

fn main() {
    duration();
}
