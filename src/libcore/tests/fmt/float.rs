#[test]
fn test_format_f64() {
    assert_eq!("1", format!("{:.0}", 1.0f64));
    assert_eq!("9", format!("{:.0}", 9.4f64));
    assert_eq!("10", format!("{:.0}", 9.9f64));
    assert_eq!("9.8", format!("{:.1}", 9.849f64));
    assert_eq!("9.9", format!("{:.1}", 9.851f64));
    assert_eq!("1", format!("{:.0}", 0.5f64));
    assert_eq!("1.23456789e6", format!("{:e}", 1234567.89f64));
    assert_eq!("1.23456789e3", format!("{:e}", 1234.56789f64));
    assert_eq!("1.23456789E6", format!("{:E}", 1234567.89f64));
    assert_eq!("1.23456789E3", format!("{:E}", 1234.56789f64));
    assert_eq!("0.0", format!("{:?}", 0.0f64));
    assert_eq!("1.01", format!("{:?}", 1.01f64));
}

#[test]
fn test_format_f32() {
    assert_eq!("1", format!("{:.0}", 1.0f32));
    assert_eq!("9", format!("{:.0}", 9.4f32));
    assert_eq!("10", format!("{:.0}", 9.9f32));
    assert_eq!("9.8", format!("{:.1}", 9.849f32));
    assert_eq!("9.9", format!("{:.1}", 9.851f32));
    assert_eq!("1", format!("{:.0}", 0.5f32));
    assert_eq!("1.2345679e6", format!("{:e}", 1234567.89f32));
    assert_eq!("1.2345679e3", format!("{:e}", 1234.56789f32));
    assert_eq!("1.2345679E6", format!("{:E}", 1234567.89f32));
    assert_eq!("1.2345679E3", format!("{:E}", 1234.56789f32));
    assert_eq!("0.0", format!("{:?}", 0.0f32));
    assert_eq!("1.01", format!("{:?}", 1.01f32));
}
