#![warn(clippy::all)]
#![allow(
    clippy::boxed_local,
    clippy::needless_pass_by_value,
    clippy::blacklisted_name,
    unused
)]

macro_rules! boxit {
    ($init:expr, $x:ty) => {
        let _: Box<$x> = Box::new($init);
    };
}

fn test_macro() {
    boxit!(Vec::new(), Vec<u8>);
}
fn test(foo: Box<Vec<bool>>) {}

fn test2(foo: Box<dyn Fn(Vec<u32>)>) {
    // pass if #31 is fixed
    foo(vec![1, 2, 3])
}

fn test_local_not_linted() {
    let _: Box<Vec<bool>>;
}

// All of these test should be allowed because they are part of the
// public api and `avoid_breaking_exported_api` is `false` by default.
pub fn pub_test(foo: Box<Vec<bool>>) {}
pub fn pub_test_ret() -> Box<Vec<bool>> {
    Box::new(Vec::new())
}

fn main() {}
