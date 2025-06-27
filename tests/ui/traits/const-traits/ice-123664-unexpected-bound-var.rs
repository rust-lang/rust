#![allow(incomplete_features)]
#![feature(generic_const_exprs, const_trait_impl)]

const fn with_positive<F: [const] Fn()>() {}
//~^ ERROR `[const]` can only be applied to `#[const_trait]` traits
//~| ERROR `[const]` can only be applied to `#[const_trait]` traits

pub fn main() {}
