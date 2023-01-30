// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![stable(feature = "foo", since = "1.33.0")]
#![feature(staged_api)]

#[stable(feature = "foo", since = "1.33.0")]
#[rustc_const_unstable(feature = "const_foo", issue = "none")]
const fn unstable(a: *const i32, b: i32) -> bool {
    *a == b
    //~^ dereference of raw pointer is unsafe
}

fn main() {}
