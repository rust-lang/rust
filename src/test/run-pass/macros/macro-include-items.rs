// run-pass
#![allow(non_camel_case_types)]

// ignore-pretty issue #37195

fn bar() {}

include!(concat!("", "", "auxiliary/", "macro-include-items-item.rs"));

fn main() {
    foo();
    assert_eq!(include!(concat!("", "auxiliary/", "macro-include-items-expr.rs")), 1_usize);
}
