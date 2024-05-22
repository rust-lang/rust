#![feature(pattern_types, rustc_attrs)]
#![feature(core_pattern_type)]
#![feature(core_pattern_types)]
#![allow(incomplete_features)]

//! check that pattern types can have traits implemented for them if
//! their base type is a local type.

use std::pat::pattern_type;

type Y = pattern_type!(u32 is 1..);

impl Y {
    //~^ ERROR cannot define inherent `impl`
    fn foo() {}
}

struct MyStruct<T>(T);

impl MyStruct<Y> {
    fn foo() {}
}

struct Wrapper(Y);

impl Wrapper {
    fn foo() {}
}

fn main() {}
