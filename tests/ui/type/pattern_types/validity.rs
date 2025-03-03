//! Check that pattern types have their validity checked

#![feature(pattern_types, const_trait_impl, pattern_type_range_trait)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

const BAD: pattern_type!(u32 is 1..) = unsafe { std::mem::transmute(0) };
//~^ ERROR: it is undefined behavior to use this value

const BAD_UNINIT: pattern_type!(u32 is 1..) =
    //~^ ERROR: evaluation of constant value failed
    unsafe { std::mem::transmute(std::mem::MaybeUninit::<u32>::uninit()) };

const BAD_PTR: pattern_type!(usize is 1..) = unsafe { std::mem::transmute(&42) };
//~^ ERROR: evaluation of constant value failed

const BAD_AGGREGATE: (pattern_type!(u32 is 1..), u32) = (unsafe { std::mem::transmute(0) }, 0);
//~^ ERROR: it is undefined behavior to use this value

struct Foo(Bar);
struct Bar(pattern_type!(u32 is 1..));

const BAD_FOO: Foo = Foo(Bar(unsafe { std::mem::transmute(0) }));
//~^ ERROR: it is undefined behavior to use this value

const CHAR_UNINIT: pattern_type!(char is 'A'..'Z') =
    //~^ ERROR: evaluation of constant value failed
    unsafe { std::mem::transmute(std::mem::MaybeUninit::<u32>::uninit()) };

const CHAR_OOB_PAT: pattern_type!(char is 'A'..'Z') = unsafe { std::mem::transmute('a') };
//~^ ERROR: it is undefined behavior to use this value

const CHAR_OOB: pattern_type!(char is 'A'..'Z') = unsafe { std::mem::transmute(u32::MAX) };
//~^ ERROR: it is undefined behavior to use this value

fn main() {}
