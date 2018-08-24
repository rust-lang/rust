use std::fmt;

#[test]
fn test_format() {
    let s = fmt::format(format_args!("Hello, {}!", "world"));
    assert_eq!(s, "Hello, world!");
}
