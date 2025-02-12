use core::time::Duration;

#[test]
fn creation() {
    assert_ne!(Duration::from_secs(1), Duration::from_secs(0));
    assert_eq!(Duration::from_secs(1) + Duration::from_secs(2), Duration::from_secs(3));
    assert_eq!(
        Duration::from_millis(10) + Duration::from_secs(4),
        Duration::new(4, 10 * 1_000_000)
    );
    assert_eq!(Duration::from_millis(4000), Duration::new(4, 0));
}

#[test]
#[should_panic]
fn new_overflow() {
    let _ = Duration::new(u64::MAX, 1_000_000_000);
}

#[test]
#[should_panic]
fn from_mins_overflow() {
    let overflow = u64::MAX / 60 + 1;
    let _ = Duration::from_mins(overflow);
}

#[test]
#[should_panic]
fn from_hours_overflow() {
    let overflow = u64::MAX / (60 * 60) + 1;
    let _ = Duration::from_hours(overflow);
}

#[test]
#[should_panic]
fn from_days_overflow() {
    let overflow = u64::MAX / (24 * 60 * 60) + 1;
    let _ = Duration::from_days(overflow);
}

#[test]
#[should_panic]
fn from_weeks_overflow() {
    let overflow = u64::MAX / (7 * 24 * 60 * 60) + 1;
    let _ = Duration::from_weeks(overflow);
}

#[test]
fn constructors() {
    assert_eq!(Duration::from_weeks(1), Duration::from_secs(7 * 24 * 60 * 60));
    assert_eq!(Duration::from_weeks(0), Duration::ZERO);

    assert_eq!(Duration::from_days(1), Duration::from_secs(86_400));
    assert_eq!(Duration::from_days(0), Duration::ZERO);

    assert_eq!(Duration::from_hours(1), Duration::from_secs(3_600));
    assert_eq!(Duration::from_hours(0), Duration::ZERO);

    assert_eq!(Duration::from_mins(1), Duration::from_secs(60));
    assert_eq!(Duration::from_mins(0), Duration::ZERO);
}

#[test]
fn secs() {
    assert_eq!(Duration::new(0, 0).as_secs(), 0);
    assert_eq!(Duration::new(0, 500_000_005).as_secs(), 0);
    assert_eq!(Duration::new(0, 1_050_000_001).as_secs(), 1);
    assert_eq!(Duration::from_secs(1).as_secs(), 1);
    assert_eq!(Duration::from_millis(999).as_secs(), 0);
    assert_eq!(Duration::from_millis(1001).as_secs(), 1);
    assert_eq!(Duration::from_micros(999_999).as_secs(), 0);
    assert_eq!(Duration::from_micros(1_000_001).as_secs(), 1);
    assert_eq!(Duration::from_nanos(999_999_999).as_secs(), 0);
    assert_eq!(Duration::from_nanos(1_000_000_001).as_secs(), 1);
}

#[test]
fn millis() {
    assert_eq!(Duration::new(0, 0).subsec_millis(), 0);
    assert_eq!(Duration::new(0, 500_000_005).subsec_millis(), 500);
    assert_eq!(Duration::new(0, 1_050_000_001).subsec_millis(), 50);
    assert_eq!(Duration::from_secs(1).subsec_millis(), 0);
    assert_eq!(Duration::from_millis(999).subsec_millis(), 999);
    assert_eq!(Duration::from_millis(1001).subsec_millis(), 1);
    assert_eq!(Duration::from_micros(999_999).subsec_millis(), 999);
    assert_eq!(Duration::from_micros(1_001_000).subsec_millis(), 1);
    assert_eq!(Duration::from_nanos(999_999_999).subsec_millis(), 999);
    assert_eq!(Duration::from_nanos(1_001_000_000).subsec_millis(), 1);
}

#[test]
fn micros() {
    assert_eq!(Duration::new(0, 0).subsec_micros(), 0);
    assert_eq!(Duration::new(0, 500_000_005).subsec_micros(), 500_000);
    assert_eq!(Duration::new(0, 1_050_000_001).subsec_micros(), 50_000);
    assert_eq!(Duration::from_secs(1).subsec_micros(), 0);
    assert_eq!(Duration::from_millis(999).subsec_micros(), 999_000);
    assert_eq!(Duration::from_millis(1001).subsec_micros(), 1_000);
    assert_eq!(Duration::from_micros(999_999).subsec_micros(), 999_999);
    assert_eq!(Duration::from_micros(1_000_001).subsec_micros(), 1);
    assert_eq!(Duration::from_nanos(999_999_999).subsec_micros(), 999_999);
    assert_eq!(Duration::from_nanos(1_000_001_000).subsec_micros(), 1);
}

#[test]
fn nanos() {
    assert_eq!(Duration::new(0, 0).subsec_nanos(), 0);
    assert_eq!(Duration::new(0, 5).subsec_nanos(), 5);
    assert_eq!(Duration::new(0, 1_000_000_001).subsec_nanos(), 1);
    assert_eq!(Duration::from_secs(1).subsec_nanos(), 0);
    assert_eq!(Duration::from_millis(999).subsec_nanos(), 999_000_000);
    assert_eq!(Duration::from_millis(1001).subsec_nanos(), 1_000_000);
    assert_eq!(Duration::from_micros(999_999).subsec_nanos(), 999_999_000);
    assert_eq!(Duration::from_micros(1_000_001).subsec_nanos(), 1000);
    assert_eq!(Duration::from_nanos(999_999_999).subsec_nanos(), 999_999_999);
    assert_eq!(Duration::from_nanos(1_000_000_001).subsec_nanos(), 1);
}

#[test]
fn abs_diff() {
    assert_eq!(Duration::new(2, 0).abs_diff(Duration::new(1, 0)), Duration::new(1, 0));
    assert_eq!(Duration::new(1, 0).abs_diff(Duration::new(2, 0)), Duration::new(1, 0));
    assert_eq!(Duration::new(1, 0).abs_diff(Duration::new(1, 0)), Duration::new(0, 0));
    assert_eq!(Duration::new(1, 1).abs_diff(Duration::new(0, 2)), Duration::new(0, 999_999_999));
    assert_eq!(Duration::new(1, 1).abs_diff(Duration::new(2, 1)), Duration::new(1, 0));
    assert_eq!(Duration::MAX.abs_diff(Duration::MAX), Duration::ZERO);
    assert_eq!(Duration::ZERO.abs_diff(Duration::ZERO), Duration::ZERO);
    assert_eq!(Duration::MAX.abs_diff(Duration::ZERO), Duration::MAX);
    assert_eq!(Duration::ZERO.abs_diff(Duration::MAX), Duration::MAX);
}

#[test]
fn add() {
    assert_eq!(Duration::new(0, 0) + Duration::new(0, 1), Duration::new(0, 1));
    assert_eq!(Duration::new(0, 500_000_000) + Duration::new(0, 500_000_001), Duration::new(1, 1));
}

#[test]
fn checked_add() {
    assert_eq!(Duration::new(0, 0).checked_add(Duration::new(0, 1)), Some(Duration::new(0, 1)));
    assert_eq!(
        Duration::new(0, 500_000_000).checked_add(Duration::new(0, 500_000_001)),
        Some(Duration::new(1, 1))
    );
    assert_eq!(Duration::new(1, 0).checked_add(Duration::new(u64::MAX, 0)), None);
}

#[test]
fn saturating_add() {
    assert_eq!(Duration::new(0, 0).saturating_add(Duration::new(0, 1)), Duration::new(0, 1));
    assert_eq!(
        Duration::new(0, 500_000_000).saturating_add(Duration::new(0, 500_000_001)),
        Duration::new(1, 1)
    );
    assert_eq!(Duration::new(1, 0).saturating_add(Duration::new(u64::MAX, 0)), Duration::MAX);
}

#[test]
fn sub() {
    assert_eq!(Duration::new(0, 1) - Duration::new(0, 0), Duration::new(0, 1));
    assert_eq!(Duration::new(0, 500_000_001) - Duration::new(0, 500_000_000), Duration::new(0, 1));
    assert_eq!(Duration::new(1, 0) - Duration::new(0, 1), Duration::new(0, 999_999_999));
}

#[test]
fn checked_sub() {
    assert_eq!(Duration::NANOSECOND.checked_sub(Duration::ZERO), Some(Duration::NANOSECOND));
    assert_eq!(
        Duration::SECOND.checked_sub(Duration::NANOSECOND),
        Some(Duration::new(0, 999_999_999))
    );
    assert_eq!(Duration::ZERO.checked_sub(Duration::NANOSECOND), None);
    assert_eq!(Duration::ZERO.checked_sub(Duration::SECOND), None);
}

#[test]
fn saturating_sub() {
    assert_eq!(Duration::NANOSECOND.saturating_sub(Duration::ZERO), Duration::NANOSECOND);
    assert_eq!(
        Duration::SECOND.saturating_sub(Duration::NANOSECOND),
        Duration::new(0, 999_999_999)
    );
    assert_eq!(Duration::ZERO.saturating_sub(Duration::NANOSECOND), Duration::ZERO);
    assert_eq!(Duration::ZERO.saturating_sub(Duration::SECOND), Duration::ZERO);
}

#[test]
#[should_panic]
fn sub_bad1() {
    let _ = Duration::new(0, 0) - Duration::new(0, 1);
}

#[test]
#[should_panic]
fn sub_bad2() {
    let _ = Duration::new(0, 0) - Duration::new(1, 0);
}

#[test]
fn mul() {
    assert_eq!(Duration::new(0, 1) * 2, Duration::new(0, 2));
    assert_eq!(Duration::new(1, 1) * 3, Duration::new(3, 3));
    assert_eq!(Duration::new(0, 500_000_001) * 4, Duration::new(2, 4));
    assert_eq!(Duration::new(0, 500_000_001) * 4000, Duration::new(2000, 4000));
}

#[test]
fn checked_mul() {
    assert_eq!(Duration::new(0, 1).checked_mul(2), Some(Duration::new(0, 2)));
    assert_eq!(Duration::new(1, 1).checked_mul(3), Some(Duration::new(3, 3)));
    assert_eq!(Duration::new(0, 500_000_001).checked_mul(4), Some(Duration::new(2, 4)));
    assert_eq!(Duration::new(0, 500_000_001).checked_mul(4000), Some(Duration::new(2000, 4000)));
    assert_eq!(Duration::new(u64::MAX - 1, 0).checked_mul(2), None);
}

#[test]
fn saturating_mul() {
    assert_eq!(Duration::new(0, 1).saturating_mul(2), Duration::new(0, 2));
    assert_eq!(Duration::new(1, 1).saturating_mul(3), Duration::new(3, 3));
    assert_eq!(Duration::new(0, 500_000_001).saturating_mul(4), Duration::new(2, 4));
    assert_eq!(Duration::new(0, 500_000_001).saturating_mul(4000), Duration::new(2000, 4000));
    assert_eq!(Duration::new(u64::MAX - 1, 0).saturating_mul(2), Duration::MAX);
}

#[test]
fn div() {
    assert_eq!(Duration::new(0, 1) / 2, Duration::new(0, 0));
    assert_eq!(Duration::new(1, 1) / 3, Duration::new(0, 333_333_333));
    assert_eq!(Duration::new(1, 1) / 7, Duration::new(0, 142_857_143));
    assert_eq!(Duration::new(99, 999_999_000) / 100, Duration::new(0, 999_999_990));
}

#[test]
fn div_duration_f32() {
    assert_eq!(Duration::ZERO.div_duration_f32(Duration::MAX), 0.0);
    assert_eq!(Duration::MAX.div_duration_f32(Duration::ZERO), f32::INFINITY);
    assert_eq!((Duration::SECOND * 2).div_duration_f32(Duration::SECOND), 2.0);
    assert!(Duration::ZERO.div_duration_f32(Duration::ZERO).is_nan());
    // These tests demonstrate it doesn't panic with extreme values.
    // Accuracy of the computed value is not a huge concern, we know floats don't work well
    // at these extremes.
    assert!((Duration::MAX).div_duration_f32(Duration::NANOSECOND) > 10.0f32.powf(28.0));
    assert!((Duration::NANOSECOND).div_duration_f32(Duration::MAX) < 0.1);
}

#[test]
fn div_duration_f64() {
    assert_eq!(Duration::ZERO.div_duration_f64(Duration::MAX), 0.0);
    assert_eq!(Duration::MAX.div_duration_f64(Duration::ZERO), f64::INFINITY);
    assert_eq!((Duration::SECOND * 2).div_duration_f64(Duration::SECOND), 2.0);
    assert!(Duration::ZERO.div_duration_f64(Duration::ZERO).is_nan());
    // These tests demonstrate it doesn't panic with extreme values.
    // Accuracy of the computed value is not a huge concern, we know floats don't work well
    // at these extremes.
    assert!((Duration::MAX).div_duration_f64(Duration::NANOSECOND) > 10.0f64.powf(28.0));
    assert!((Duration::NANOSECOND).div_duration_f64(Duration::MAX) < 0.1);
}

#[test]
fn checked_div() {
    assert_eq!(Duration::new(2, 0).checked_div(2), Some(Duration::new(1, 0)));
    assert_eq!(Duration::new(1, 0).checked_div(2), Some(Duration::new(0, 500_000_000)));
    assert_eq!(Duration::new(2, 0).checked_div(0), None);
}

#[test]
fn correct_sum() {
    let durations = [
        Duration::new(1, 999_999_999),
        Duration::new(2, 999_999_999),
        Duration::new(0, 999_999_999),
        Duration::new(0, 999_999_999),
        Duration::new(0, 999_999_999),
        Duration::new(5, 0),
    ];
    let sum = durations.iter().sum::<Duration>();
    assert_eq!(sum, Duration::new(1 + 2 + 5 + 4, 1_000_000_000 - 5));
}

#[test]
fn debug_formatting_extreme_values() {
    assert_eq!(
        format!("{:?}", Duration::new(u64::MAX, 123_456_789)),
        "18446744073709551615.123456789s"
    );
    assert_eq!(format!("{:.0?}", Duration::MAX), "18446744073709551616s");
    assert_eq!(format!("{:.0?}", Duration::new(u64::MAX, 500_000_000)), "18446744073709551616s");
    assert_eq!(format!("{:.0?}", Duration::new(u64::MAX, 499_999_999)), "18446744073709551615s");
    assert_eq!(
        format!("{:.3?}", Duration::new(u64::MAX, 999_500_000)),
        "18446744073709551616.000s"
    );
    assert_eq!(
        format!("{:.3?}", Duration::new(u64::MAX, 999_499_999)),
        "18446744073709551615.999s"
    );
    assert_eq!(
        format!("{:.8?}", Duration::new(u64::MAX, 999_999_995)),
        "18446744073709551616.00000000s"
    );
    assert_eq!(
        format!("{:.8?}", Duration::new(u64::MAX, 999_999_994)),
        "18446744073709551615.99999999s"
    );
    assert_eq!(format!("{:21.0?}", Duration::MAX), "18446744073709551616s");
    assert_eq!(format!("{:22.0?}", Duration::MAX), "18446744073709551616s ");
    assert_eq!(format!("{:24.0?}", Duration::MAX), "18446744073709551616s   ");
}

#[test]
fn debug_formatting_secs() {
    assert_eq!(format!("{:?}", Duration::new(7, 000_000_000)), "7s");
    assert_eq!(format!("{:?}", Duration::new(7, 100_000_000)), "7.1s");
    assert_eq!(format!("{:?}", Duration::new(7, 000_010_000)), "7.00001s");
    assert_eq!(format!("{:?}", Duration::new(7, 000_000_001)), "7.000000001s");
    assert_eq!(format!("{:?}", Duration::new(7, 123_456_789)), "7.123456789s");

    assert_eq!(format!("{:?}", Duration::new(88, 000_000_000)), "88s");
    assert_eq!(format!("{:?}", Duration::new(88, 100_000_000)), "88.1s");
    assert_eq!(format!("{:?}", Duration::new(88, 000_010_000)), "88.00001s");
    assert_eq!(format!("{:?}", Duration::new(88, 000_000_001)), "88.000000001s");
    assert_eq!(format!("{:?}", Duration::new(88, 123_456_789)), "88.123456789s");

    assert_eq!(format!("{:?}", Duration::new(999, 000_000_000)), "999s");
    assert_eq!(format!("{:?}", Duration::new(999, 100_000_000)), "999.1s");
    assert_eq!(format!("{:?}", Duration::new(999, 000_010_000)), "999.00001s");
    assert_eq!(format!("{:?}", Duration::new(999, 000_000_001)), "999.000000001s");
    assert_eq!(format!("{:?}", Duration::new(999, 123_456_789)), "999.123456789s");
}

#[test]
fn debug_formatting_millis() {
    assert_eq!(format!("{:?}", Duration::new(0, 7_000_000)), "7ms");
    assert_eq!(format!("{:?}", Duration::new(0, 7_100_000)), "7.1ms");
    assert_eq!(format!("{:?}", Duration::new(0, 7_000_001)), "7.000001ms");
    assert_eq!(format!("{:?}", Duration::new(0, 7_123_456)), "7.123456ms");

    assert_eq!(format!("{:?}", Duration::new(0, 88_000_000)), "88ms");
    assert_eq!(format!("{:?}", Duration::new(0, 88_100_000)), "88.1ms");
    assert_eq!(format!("{:?}", Duration::new(0, 88_000_001)), "88.000001ms");
    assert_eq!(format!("{:?}", Duration::new(0, 88_123_456)), "88.123456ms");

    assert_eq!(format!("{:?}", Duration::new(0, 999_000_000)), "999ms");
    assert_eq!(format!("{:?}", Duration::new(0, 999_100_000)), "999.1ms");
    assert_eq!(format!("{:?}", Duration::new(0, 999_000_001)), "999.000001ms");
    assert_eq!(format!("{:?}", Duration::new(0, 999_123_456)), "999.123456ms");
}

#[test]
fn debug_formatting_micros() {
    assert_eq!(format!("{:?}", Duration::new(0, 7_000)), "7µs");
    assert_eq!(format!("{:?}", Duration::new(0, 7_100)), "7.1µs");
    assert_eq!(format!("{:?}", Duration::new(0, 7_001)), "7.001µs");
    assert_eq!(format!("{:?}", Duration::new(0, 7_123)), "7.123µs");

    assert_eq!(format!("{:?}", Duration::new(0, 88_000)), "88µs");
    assert_eq!(format!("{:?}", Duration::new(0, 88_100)), "88.1µs");
    assert_eq!(format!("{:?}", Duration::new(0, 88_001)), "88.001µs");
    assert_eq!(format!("{:?}", Duration::new(0, 88_123)), "88.123µs");

    assert_eq!(format!("{:?}", Duration::new(0, 999_000)), "999µs");
    assert_eq!(format!("{:?}", Duration::new(0, 999_100)), "999.1µs");
    assert_eq!(format!("{:?}", Duration::new(0, 999_001)), "999.001µs");
    assert_eq!(format!("{:?}", Duration::new(0, 999_123)), "999.123µs");
}

#[test]
fn debug_formatting_nanos() {
    assert_eq!(format!("{:?}", Duration::new(0, 0)), "0ns");
    assert_eq!(format!("{:?}", Duration::new(0, 1)), "1ns");
    assert_eq!(format!("{:?}", Duration::new(0, 88)), "88ns");
    assert_eq!(format!("{:?}", Duration::new(0, 999)), "999ns");
}

#[test]
fn debug_formatting_precision_zero() {
    assert_eq!(format!("{:.0?}", Duration::new(0, 0)), "0ns");
    assert_eq!(format!("{:.0?}", Duration::new(0, 123)), "123ns");

    assert_eq!(format!("{:.0?}", Duration::new(0, 1_001)), "1µs");
    assert_eq!(format!("{:.0?}", Duration::new(0, 1_499)), "1µs");
    assert_eq!(format!("{:.0?}", Duration::new(0, 1_500)), "2µs");
    assert_eq!(format!("{:.0?}", Duration::new(0, 1_999)), "2µs");

    assert_eq!(format!("{:.0?}", Duration::new(0, 1_000_001)), "1ms");
    assert_eq!(format!("{:.0?}", Duration::new(0, 1_499_999)), "1ms");
    assert_eq!(format!("{:.0?}", Duration::new(0, 1_500_000)), "2ms");
    assert_eq!(format!("{:.0?}", Duration::new(0, 1_999_999)), "2ms");

    assert_eq!(format!("{:.0?}", Duration::new(1, 000_000_001)), "1s");
    assert_eq!(format!("{:.0?}", Duration::new(1, 499_999_999)), "1s");
    assert_eq!(format!("{:.0?}", Duration::new(1, 500_000_000)), "2s");
    assert_eq!(format!("{:.0?}", Duration::new(1, 999_999_999)), "2s");
}

#[test]
fn debug_formatting_precision_two() {
    assert_eq!(format!("{:.2?}", Duration::new(0, 0)), "0.00ns");
    assert_eq!(format!("{:.2?}", Duration::new(0, 123)), "123.00ns");

    assert_eq!(format!("{:.2?}", Duration::new(0, 1_000)), "1.00µs");
    assert_eq!(format!("{:.2?}", Duration::new(0, 7_001)), "7.00µs");
    assert_eq!(format!("{:.2?}", Duration::new(0, 7_100)), "7.10µs");
    assert_eq!(format!("{:.2?}", Duration::new(0, 7_109)), "7.11µs");
    assert_eq!(format!("{:.2?}", Duration::new(0, 7_199)), "7.20µs");
    assert_eq!(format!("{:.2?}", Duration::new(0, 1_999)), "2.00µs");

    assert_eq!(format!("{:.2?}", Duration::new(0, 1_000_000)), "1.00ms");
    assert_eq!(format!("{:.2?}", Duration::new(0, 3_001_000)), "3.00ms");
    assert_eq!(format!("{:.2?}", Duration::new(0, 3_100_000)), "3.10ms");
    assert_eq!(format!("{:.2?}", Duration::new(0, 1_999_999)), "2.00ms");

    assert_eq!(format!("{:.2?}", Duration::new(1, 000_000_000)), "1.00s");
    assert_eq!(format!("{:.2?}", Duration::new(4, 001_000_000)), "4.00s");
    assert_eq!(format!("{:.2?}", Duration::new(2, 100_000_000)), "2.10s");
    assert_eq!(format!("{:.2?}", Duration::new(2, 104_990_000)), "2.10s");
    assert_eq!(format!("{:.2?}", Duration::new(2, 105_000_000)), "2.11s");
    assert_eq!(format!("{:.2?}", Duration::new(8, 999_999_999)), "9.00s");
}

#[test]
fn debug_formatting_padding() {
    assert_eq!("0ns      ", format!("{:<9?}", Duration::new(0, 0)));
    assert_eq!("      0ns", format!("{:>9?}", Duration::new(0, 0)));
    assert_eq!("   0ns   ", format!("{:^9?}", Duration::new(0, 0)));
    assert_eq!("123ns    ", format!("{:<9.0?}", Duration::new(0, 123)));
    assert_eq!("    123ns", format!("{:>9.0?}", Duration::new(0, 123)));
    assert_eq!("  123ns  ", format!("{:^9.0?}", Duration::new(0, 123)));
    assert_eq!("123.0ns  ", format!("{:<9.1?}", Duration::new(0, 123)));
    assert_eq!("  123.0ns", format!("{:>9.1?}", Duration::new(0, 123)));
    assert_eq!(" 123.0ns ", format!("{:^9.1?}", Duration::new(0, 123)));
    assert_eq!("7.1µs    ", format!("{:<9?}", Duration::new(0, 7_100)));
    assert_eq!("    7.1µs", format!("{:>9?}", Duration::new(0, 7_100)));
    assert_eq!("  7.1µs  ", format!("{:^9?}", Duration::new(0, 7_100)));
    assert_eq!("999.123456ms", format!("{:<9?}", Duration::new(0, 999_123_456)));
    assert_eq!("999.123456ms", format!("{:>9?}", Duration::new(0, 999_123_456)));
    assert_eq!("999.123456ms", format!("{:^9?}", Duration::new(0, 999_123_456)));
    assert_eq!("5s       ", format!("{:<9?}", Duration::new(5, 0)));
    assert_eq!("       5s", format!("{:>9?}", Duration::new(5, 0)));
    assert_eq!("   5s    ", format!("{:^9?}", Duration::new(5, 0)));
    assert_eq!("5.000000000000s", format!("{:<9.12?}", Duration::new(5, 0)));
    assert_eq!("5.000000000000s", format!("{:>9.12?}", Duration::new(5, 0)));
    assert_eq!("5.000000000000s", format!("{:^9.12?}", Duration::new(5, 0)));

    // default alignment is left:
    assert_eq!("5s       ", format!("{:9?}", Duration::new(5, 0)));
}

#[test]
fn debug_formatting_precision_high() {
    assert_eq!(format!("{:.5?}", Duration::new(0, 23_678)), "23.67800µs");

    assert_eq!(format!("{:.9?}", Duration::new(1, 000_000_000)), "1.000000000s");
    assert_eq!(format!("{:.10?}", Duration::new(4, 001_000_000)), "4.0010000000s");
    assert_eq!(format!("{:.20?}", Duration::new(4, 001_000_000)), "4.00100000000000000000s");
}

#[test]
fn duration_const() {
    // test that the methods of `Duration` are usable in a const context

    const DURATION: Duration = Duration::new(0, 123_456_789);

    const SUB_SEC_MILLIS: u32 = DURATION.subsec_millis();
    assert_eq!(SUB_SEC_MILLIS, 123);

    const SUB_SEC_MICROS: u32 = DURATION.subsec_micros();
    assert_eq!(SUB_SEC_MICROS, 123_456);

    const SUB_SEC_NANOS: u32 = DURATION.subsec_nanos();
    assert_eq!(SUB_SEC_NANOS, 123_456_789);

    const IS_ZERO: bool = Duration::ZERO.is_zero();
    assert!(IS_ZERO);

    const SECONDS: u64 = Duration::SECOND.as_secs();
    assert_eq!(SECONDS, 1);

    const FROM_SECONDS: Duration = Duration::from_secs(1);
    assert_eq!(FROM_SECONDS, Duration::SECOND);

    const SECONDS_F32: f32 = Duration::SECOND.as_secs_f32();
    assert_eq!(SECONDS_F32, 1.0);

    // FIXME(#110395)
    // const FROM_SECONDS_F32: Duration = Duration::from_secs_f32(1.0);
    // assert_eq!(FROM_SECONDS_F32, Duration::SECOND);

    const SECONDS_F64: f64 = Duration::SECOND.as_secs_f64();
    assert_eq!(SECONDS_F64, 1.0);

    // FIXME(#110395)
    // const FROM_SECONDS_F64: Duration = Duration::from_secs_f64(1.0);
    // assert_eq!(FROM_SECONDS_F64, Duration::SECOND);

    const MILLIS: u128 = Duration::SECOND.as_millis();
    assert_eq!(MILLIS, 1_000);

    const FROM_MILLIS: Duration = Duration::from_millis(1_000);
    assert_eq!(FROM_MILLIS, Duration::SECOND);

    const MICROS: u128 = Duration::SECOND.as_micros();
    assert_eq!(MICROS, 1_000_000);

    const FROM_MICROS: Duration = Duration::from_micros(1_000_000);
    assert_eq!(FROM_MICROS, Duration::SECOND);

    const NANOS: u128 = Duration::SECOND.as_nanos();
    assert_eq!(NANOS, 1_000_000_000);

    const FROM_NANOS: Duration = Duration::from_nanos(1_000_000_000);
    assert_eq!(FROM_NANOS, Duration::SECOND);

    const MAX: Duration = Duration::new(u64::MAX, 999_999_999);

    const CHECKED_ADD: Option<Duration> = MAX.checked_add(Duration::SECOND);
    assert_eq!(CHECKED_ADD, None);

    const CHECKED_SUB: Option<Duration> = Duration::ZERO.checked_sub(Duration::SECOND);
    assert_eq!(CHECKED_SUB, None);

    const CHECKED_MUL: Option<Duration> = Duration::SECOND.checked_mul(1);
    assert_eq!(CHECKED_MUL, Some(Duration::SECOND));

    /*  FIXME(#110395)
        const MUL_F32: Duration = Duration::SECOND.mul_f32(1.0);
        assert_eq!(MUL_F32, Duration::SECOND);

        const MUL_F64: Duration = Duration::SECOND.mul_f64(1.0);
        assert_eq!(MUL_F64, Duration::SECOND);

        const CHECKED_DIV: Option<Duration> = Duration::SECOND.checked_div(1);
        assert_eq!(CHECKED_DIV, Some(Duration::SECOND));

        const DIV_F32: Duration = Duration::SECOND.div_f32(1.0);
        assert_eq!(DIV_F32, Duration::SECOND);

        const DIV_F64: Duration = Duration::SECOND.div_f64(1.0);
        assert_eq!(DIV_F64, Duration::SECOND);
    */

    const DIV_DURATION_F32: f32 = Duration::SECOND.div_duration_f32(Duration::SECOND);
    assert_eq!(DIV_DURATION_F32, 1.0);

    const DIV_DURATION_F64: f64 = Duration::SECOND.div_duration_f64(Duration::SECOND);
    assert_eq!(DIV_DURATION_F64, 1.0);

    const SATURATING_ADD: Duration = MAX.saturating_add(Duration::SECOND);
    assert_eq!(SATURATING_ADD, MAX);

    const SATURATING_SUB: Duration = Duration::ZERO.saturating_sub(Duration::SECOND);
    assert_eq!(SATURATING_SUB, Duration::ZERO);

    const SATURATING_MUL: Duration = MAX.saturating_mul(2);
    assert_eq!(SATURATING_MUL, MAX);
}

#[test]
fn from_neg_zero() {
    assert_eq!(Duration::try_from_secs_f32(-0.0), Ok(Duration::ZERO));
    assert_eq!(Duration::try_from_secs_f64(-0.0), Ok(Duration::ZERO));
    assert_eq!(Duration::from_secs_f32(-0.0), Duration::ZERO);
    assert_eq!(Duration::from_secs_f64(-0.0), Duration::ZERO);
}
