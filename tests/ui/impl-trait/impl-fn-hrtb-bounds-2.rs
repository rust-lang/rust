//@revisions: edition2015 edition2024
//@[edition2015] edition:2015
//@[edition2024] edition:2024
#![feature(impl_trait_in_fn_trait_return)]
use std::fmt::Debug;

fn a() -> impl Fn(&u8) -> impl Debug { //[edition2024]~ ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`
    |x| x //[edition2015]~ ERROR hidden type for `impl Debug` captures lifetime that does not appear in bounds
}

fn main() {}
