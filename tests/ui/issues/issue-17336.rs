//@ build-pass

#![allow(unused_must_use)]
#![allow(ambiguous_wide_pointer_comparisons)]

#[allow(dead_code)]
fn check(a: &str) {
    let x = a as *const str;
    x == x;
}

fn main() {}
