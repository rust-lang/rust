#![feature(impl_trait_in_fn_trait_return)]
use std::fmt::Debug;

fn a() -> impl Fn(&u8) -> impl Debug {
    |x| x //~ ERROR hidden type for `impl Debug` captures lifetime that does not appear in bounds
}

fn main() {}
