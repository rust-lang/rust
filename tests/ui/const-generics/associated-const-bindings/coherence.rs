#![feature(gca_min_const_items)]
#![expect(incomplete_features)]

use std::gca;

pub trait IsVoid {
    #[rustc_always_gca]
    const IS_VOID: bool;
}
impl IsVoid for () {
    const IS_VOID: bool = gca!(true);
}

pub trait Maybe {}
impl Maybe for () {}
impl Maybe for () where (): IsVoid<IS_VOID = true> {}
//~^ ERROR conflicting implementations of trait `Maybe` for type `()`

fn main() {}
