mod builders;
mod float;
mod num;

#[test]
fn test_format_flags() {
    // No residual flags left by pointer formatting
    let p = "".as_ptr();
    assert_eq!(format!("{:p} {:x}", p, 16), format!("{:p} 10", p));

    assert_eq!(format!("{: >3}", 'a'), "  a");
}

#[test]
fn test_pointer_formats_data_pointer() {
    let b: &[u8] = b"";
    let s: &str = "";
    assert_eq!(format!("{:p}", s), format!("{:p}", s.as_ptr()));
    assert_eq!(format!("{:p}", b), format!("{:p}", b.as_ptr()));
}

#[test]
fn test_estimated_capacity() {
    assert_eq!(format_args!("").estimated_capacity(), 0);
    assert_eq!(format_args!("{}", "").estimated_capacity(), 0);
    assert_eq!(format_args!("Hello").estimated_capacity(), 5);
    assert_eq!(format_args!("Hello, {}!", "").estimated_capacity(), 16);
    assert_eq!(format_args!("{}, hello!", "World").estimated_capacity(), 0);
    assert_eq!(format_args!("{}. 16-bytes piece", "World").estimated_capacity(), 32);
}
