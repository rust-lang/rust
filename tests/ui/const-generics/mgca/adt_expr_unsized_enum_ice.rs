#![feature(gca_min_const_items, min_adt_const_params, gca_adts)]

use std::gca;

enum Foo {
    A(str),
    //~^ ERROR size for values of type `str` cannot be known at compilation time
    B,
    C(usize),
    D { field: usize },
}
impl Foo {
    const UNIT: Self = gca!(Self::B);
    //~^ ERROR must implement `ConstParamTy`
    const TUPLE: Self = gca!(Self::C(2));
    //~^ ERROR must implement `ConstParamTy`
    const STRUCT: Self = gca!(Self::D { field: 3 });
    //~^ ERROR must implement `ConstParamTy`
}

fn main() {
    Foo::UNIT;
    Foo::TUPLE;
    Foo::STRUCT;
}
