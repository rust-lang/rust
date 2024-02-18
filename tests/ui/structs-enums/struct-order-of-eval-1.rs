//@ run-pass
#![allow(dead_code)]

struct S { f0: String, f1: isize }

pub fn main() {
    let s = "Hello, world!".to_string();
    let s = S {
        f0: s.to_string(),
        ..S {
            f0: s,
            f1: 23
        }
    };
    assert_eq!(s.f0, "Hello, world!");
}
