#![allow(clippy::redundant_clone)]
#![feature(custom_inner_attributes)]
#![clippy::msrv = "1.0"]

fn manual_strip_msrv() {
    let s = "hello, world!";
    if s.starts_with("hello, ") {
        assert_eq!(s["hello, ".len()..].to_uppercase(), "WORLD!");
    }
}

fn main() {
    manual_strip_msrv()
}
