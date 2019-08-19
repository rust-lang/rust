// build-pass (FIXME(62277): could be check-pass?)
#![allow(unused_must_use)]
#[allow(dead_code)]
fn check(a: &str) {
    let x = a as *const str;
    x == x;
}

fn main() {}
