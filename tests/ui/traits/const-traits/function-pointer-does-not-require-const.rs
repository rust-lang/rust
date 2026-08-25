//@ check-pass
#![feature(const_trait_impl)]

pub const trait Test {}

impl Test for () {}

pub const fn test<T: [const] Test>() {}

pub const fn min_by_i32() -> fn() {
    test::<()>
}

fn main() {}
