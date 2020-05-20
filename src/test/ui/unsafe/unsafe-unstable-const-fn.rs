#![stable(feature = "foo", since = "1.33.0")]
#![feature(staged_api)]
#![feature(const_compare_raw_pointers)]
#![feature(const_fn)]

#[stable(feature = "foo", since = "1.33.0")]
#[rustc_const_unstable(feature = "const_foo", issue = "none")]
const fn unstable(a: *const i32, b: *const i32) -> bool {
    a == b
    //~^ pointer operation is unsafe
}

fn main() {}
