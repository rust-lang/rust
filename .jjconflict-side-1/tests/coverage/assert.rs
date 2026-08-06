#![feature(coverage_attribute)]
//@ edition: 2024

use core::assert_matches;

fn assert_a_plain(opt: Option<&str>) {
    assert!(opt.is_some());
    assert_eq!(opt, Some("true"));
    assert_ne!(opt, None);
    assert_matches!(opt, Some("true"));
}

fn assert_b_message(opt: Option<&str>) {
    assert!(opt.is_some(), "message");
    assert_eq!(opt, Some("true"), "message");
    assert_ne!(opt, None, "message");
    assert_matches!(opt, Some("true"), "message");
}

fn assert_c_format_inline(opt: Option<&str>, msg: &str) {
    assert!(opt.is_some(), "message: {msg}");
    assert_eq!(opt, Some("true"), "message: {msg}");
    assert_ne!(opt, None, "message: {msg}");
    assert_matches!(opt, Some("true"), "message: {msg}");
}

fn assert_d_format_arg(opt: Option<&str>, msg: &str) {
    assert!(opt.is_some(), "message: {}", msg);
    assert_eq!(opt, Some("true"), "message: {}", msg);
    assert_ne!(opt, None, "message: {}", msg);
    assert_matches!(opt, Some("true"), "message: {}", msg);
}

#[coverage(off)]
fn main() {
    let opt = core::hint::black_box(Some("true"));
    assert_a_plain(opt);
    assert_b_message(opt);
    assert_c_format_inline(opt, "message");
    assert_d_format_arg(opt, "message");
}
