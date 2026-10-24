//@ failure-status: 1
//@ check-fail
//@ compile-flags: --emit=mir
#![feature(gca_min_const_items, min_adt_const_params, gca_adts)]

use std::gca;

enum Foo {
    A(str),
    //~^ ERROR size for values of type `str` cannot be known at compilation time
    B,
}
impl Foo {
    const ASSOC: Self = gca!(Self::B);
    //~^ ERROR must implement `ConstParamTy`
}

fn main() {
    Foo::ASSOC;
}
