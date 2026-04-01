#[test]
fn test_format_f64() {
    assert_eq!("1", format!("{:.0}", 1.0f64));
    assert_eq!("9", format!("{:.0}", 9.4f64));
    assert_eq!("10", format!("{:.0}", 9.9f64));
    assert_eq!("9.8", format!("{:.1}", 9.849f64));
    assert_eq!("9.9", format!("{:.1}", 9.851f64));
    assert_eq!("0", format!("{:.0}", 0.5f64));
    assert_eq!("1.23456789e6", format!("{:e}", 1234567.89f64));
    assert_eq!("1.23456789e3", format!("{:e}", 1234.56789f64));
    assert_eq!("1.23456789E6", format!("{:E}", 1234567.89f64));
    assert_eq!("1.23456789E3", format!("{:E}", 1234.56789f64));
    assert_eq!("0.0", format!("{:?}", 0.0f64));
    assert_eq!("1.01", format!("{:?}", 1.01f64));

    let high_cutoff = 1e16_f64;
    assert_eq!("1e16", format!("{:?}", high_cutoff));
    assert_eq!("-1e16", format!("{:?}", -high_cutoff));
    assert!(!is_exponential(&format!("{:?}", high_cutoff * (1.0 - 2.0 * f64::EPSILON))));
    assert_eq!("-3.0", format!("{:?}", -3f64));
    assert_eq!("0.0001", format!("{:?}", 0.0001f64));
    assert_eq!("9e-5", format!("{:?}", 0.00009f64));
    assert_eq!("1234567.9", format!("{:.1?}", 1234567.89f64));
    assert_eq!("1234.6", format!("{:.1?}", 1234.56789f64));
}

#[test]
fn test_format_f64_rounds_ties_to_even() {
    assert_eq!("0", format!("{:.0}", 0.5f64));
    assert_eq!("2", format!("{:.0}", 1.5f64));
    assert_eq!("2", format!("{:.0}", 2.5f64));
    assert_eq!("4", format!("{:.0}", 3.5f64));
    assert_eq!("4", format!("{:.0}", 4.5f64));
    assert_eq!("6", format!("{:.0}", 5.5f64));
    assert_eq!("128", format!("{:.0}", 127.5f64));
    assert_eq!("128", format!("{:.0}", 128.5f64));
    assert_eq!("0.2", format!("{:.1}", 0.25f64));
    assert_eq!("0.8", format!("{:.1}", 0.75f64));
    assert_eq!("0.12", format!("{:.2}", 0.125f64));
    assert_eq!("0.88", format!("{:.2}", 0.875f64));
    assert_eq!("0.062", format!("{:.3}", 0.062f64));
    assert_eq!("-0", format!("{:.0}", -0.5f64));
    assert_eq!("-2", format!("{:.0}", -1.5f64));
    assert_eq!("-2", format!("{:.0}", -2.5f64));
    assert_eq!("-4", format!("{:.0}", -3.5f64));
    assert_eq!("-4", format!("{:.0}", -4.5f64));
    assert_eq!("-6", format!("{:.0}", -5.5f64));
    assert_eq!("-128", format!("{:.0}", -127.5f64));
    assert_eq!("-128", format!("{:.0}", -128.5f64));
    assert_eq!("-0.2", format!("{:.1}", -0.25f64));
    assert_eq!("-0.8", format!("{:.1}", -0.75f64));
    assert_eq!("-0.12", format!("{:.2}", -0.125f64));
    assert_eq!("-0.88", format!("{:.2}", -0.875f64));
    assert_eq!("-0.062", format!("{:.3}", -0.062f64));

    assert_eq!("2e0", format!("{:.0e}", 1.5f64));
    assert_eq!("2e0", format!("{:.0e}", 2.5f64));
    assert_eq!("4e0", format!("{:.0e}", 3.5f64));
    assert_eq!("4e0", format!("{:.0e}", 4.5f64));
    assert_eq!("6e0", format!("{:.0e}", 5.5f64));
    assert_eq!("1.28e2", format!("{:.2e}", 127.5f64));
    assert_eq!("1.28e2", format!("{:.2e}", 128.5f64));
    assert_eq!("-2e0", format!("{:.0e}", -1.5f64));
    assert_eq!("-2e0", format!("{:.0e}", -2.5f64));
    assert_eq!("-4e0", format!("{:.0e}", -3.5f64));
    assert_eq!("-4e0", format!("{:.0e}", -4.5f64));
    assert_eq!("-6e0", format!("{:.0e}", -5.5f64));
    assert_eq!("-1.28e2", format!("{:.2e}", -127.5f64));
    assert_eq!("-1.28e2", format!("{:.2e}", -128.5f64));

    assert_eq!("2E0", format!("{:.0E}", 1.5f64));
    assert_eq!("2E0", format!("{:.0E}", 2.5f64));
    assert_eq!("4E0", format!("{:.0E}", 3.5f64));
    assert_eq!("4E0", format!("{:.0E}", 4.5f64));
    assert_eq!("6E0", format!("{:.0E}", 5.5f64));
    assert_eq!("1.28E2", format!("{:.2E}", 127.5f64));
    assert_eq!("1.28E2", format!("{:.2E}", 128.5f64));
    assert_eq!("-2E0", format!("{:.0E}", -1.5f64));
    assert_eq!("-2E0", format!("{:.0E}", -2.5f64));
    assert_eq!("-4E0", format!("{:.0E}", -3.5f64));
    assert_eq!("-4E0", format!("{:.0E}", -4.5f64));
    assert_eq!("-6E0", format!("{:.0E}", -5.5f64));
    assert_eq!("-1.28E2", format!("{:.2E}", -127.5f64));
    assert_eq!("-1.28E2", format!("{:.2E}", -128.5f64));
}

#[test]
fn test_format_f32() {
    assert_eq!("1", format!("{:.0}", 1.0f32));
    assert_eq!("9", format!("{:.0}", 9.4f32));
    assert_eq!("10", format!("{:.0}", 9.9f32));
    assert_eq!("9.8", format!("{:.1}", 9.849f32));
    assert_eq!("9.9", format!("{:.1}", 9.851f32));
    assert_eq!("0", format!("{:.0}", 0.5f32));
    assert_eq!("1.2345679e6", format!("{:e}", 1234567.89f32));
    assert_eq!("1.2345679e3", format!("{:e}", 1234.56789f32));
    assert_eq!("1.2345679E6", format!("{:E}", 1234567.89f32));
    assert_eq!("1.2345679E3", format!("{:E}", 1234.56789f32));
    assert_eq!("0.0", format!("{:?}", 0.0f32));
    assert_eq!("1.01", format!("{:?}", 1.01f32));

    let high_cutoff = 1e16_f32;
    assert_eq!("1e16", format!("{:?}", high_cutoff));
    assert_eq!("-1e16", format!("{:?}", -high_cutoff));
    assert!(!is_exponential(&format!("{:?}", high_cutoff * (1.0 - 2.0 * f32::EPSILON))));
    assert_eq!("-3.0", format!("{:?}", -3f32));
    assert_eq!("0.0001", format!("{:?}", 0.0001f32));
    assert_eq!("9e-5", format!("{:?}", 0.00009f32));
    assert_eq!("1234567.9", format!("{:.1?}", 1234567.89f32));
    assert_eq!("1234.6", format!("{:.1?}", 1234.56789f32));
}

#[test]
fn test_format_f32_rounds_ties_to_even() {
    assert_eq!("0", format!("{:.0}", 0.5f32));
    assert_eq!("2", format!("{:.0}", 1.5f32));
    assert_eq!("2", format!("{:.0}", 2.5f32));
    assert_eq!("4", format!("{:.0}", 3.5f32));
    assert_eq!("4", format!("{:.0}", 4.5f32));
    assert_eq!("6", format!("{:.0}", 5.5f32));
    assert_eq!("128", format!("{:.0}", 127.5f32));
    assert_eq!("128", format!("{:.0}", 128.5f32));
    assert_eq!("0.2", format!("{:.1}", 0.25f32));
    assert_eq!("0.8", format!("{:.1}", 0.75f32));
    assert_eq!("0.12", format!("{:.2}", 0.125f32));
    assert_eq!("0.88", format!("{:.2}", 0.875f32));
    assert_eq!("0.062", format!("{:.3}", 0.062f32));
    assert_eq!("-0", format!("{:.0}", -0.5f32));
    assert_eq!("-2", format!("{:.0}", -1.5f32));
    assert_eq!("-2", format!("{:.0}", -2.5f32));
    assert_eq!("-4", format!("{:.0}", -3.5f32));
    assert_eq!("-4", format!("{:.0}", -4.5f32));
    assert_eq!("-6", format!("{:.0}", -5.5f32));
    assert_eq!("-128", format!("{:.0}", -127.5f32));
    assert_eq!("-128", format!("{:.0}", -128.5f32));
    assert_eq!("-0.2", format!("{:.1}", -0.25f32));
    assert_eq!("-0.8", format!("{:.1}", -0.75f32));
    assert_eq!("-0.12", format!("{:.2}", -0.125f32));
    assert_eq!("-0.88", format!("{:.2}", -0.875f32));
    assert_eq!("-0.062", format!("{:.3}", -0.062f32));

    assert_eq!("2e0", format!("{:.0e}", 1.5f32));
    assert_eq!("2e0", format!("{:.0e}", 2.5f32));
    assert_eq!("4e0", format!("{:.0e}", 3.5f32));
    assert_eq!("4e0", format!("{:.0e}", 4.5f32));
    assert_eq!("6e0", format!("{:.0e}", 5.5f32));
    assert_eq!("1.28e2", format!("{:.2e}", 127.5f32));
    assert_eq!("1.28e2", format!("{:.2e}", 128.5f32));
    assert_eq!("-2e0", format!("{:.0e}", -1.5f32));
    assert_eq!("-2e0", format!("{:.0e}", -2.5f32));
    assert_eq!("-4e0", format!("{:.0e}", -3.5f32));
    assert_eq!("-4e0", format!("{:.0e}", -4.5f32));
    assert_eq!("-6e0", format!("{:.0e}", -5.5f32));
    assert_eq!("-1.28e2", format!("{:.2e}", -127.5f32));
    assert_eq!("-1.28e2", format!("{:.2e}", -128.5f32));

    assert_eq!("2E0", format!("{:.0E}", 1.5f32));
    assert_eq!("2E0", format!("{:.0E}", 2.5f32));
    assert_eq!("4E0", format!("{:.0E}", 3.5f32));
    assert_eq!("4E0", format!("{:.0E}", 4.5f32));
    assert_eq!("6E0", format!("{:.0E}", 5.5f32));
    assert_eq!("1.28E2", format!("{:.2E}", 127.5f32));
    assert_eq!("1.28E2", format!("{:.2E}", 128.5f32));
    assert_eq!("-2E0", format!("{:.0E}", -1.5f32));
    assert_eq!("-2E0", format!("{:.0E}", -2.5f32));
    assert_eq!("-4E0", format!("{:.0E}", -3.5f32));
    assert_eq!("-4E0", format!("{:.0E}", -4.5f32));
    assert_eq!("-6E0", format!("{:.0E}", -5.5f32));
    assert_eq!("-1.28E2", format!("{:.2E}", -127.5f32));
    assert_eq!("-1.28E2", format!("{:.2E}", -128.5f32));
}

fn is_exponential(s: &str) -> bool {
    s.contains("e") || s.contains("E")
}
