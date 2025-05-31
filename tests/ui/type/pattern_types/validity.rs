//! Check that pattern types have their validity checked
// Strip out raw byte dumps to make tests platform-independent:
//@ normalize-stderr: "([[:xdigit:]]{2}\s){4,8}\s+│\s.{4,8}" -> "HEX_DUMP"

#![feature(pattern_types, const_trait_impl, pattern_type_range_trait)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

const BAD: pattern_type!(u32 is 1..) = unsafe { std::mem::transmute(0) };
//~^ ERROR: constructing invalid value: encountered 0

const BAD_UNINIT: pattern_type!(u32 is 1..) =
    //~^ ERROR: using uninitialized data, but this operation requires initialized memory
    unsafe { std::mem::transmute(std::mem::MaybeUninit::<u32>::uninit()) };

const BAD_PTR: pattern_type!(usize is 1..) = unsafe { std::mem::transmute(&42) };
//~^ ERROR: unable to turn pointer into integer

const BAD_AGGREGATE: (pattern_type!(u32 is 1..), u32) = (unsafe { std::mem::transmute(0) }, 0);
//~^ ERROR: constructing invalid value at .0: encountered 0

struct Foo(Bar);
struct Bar(pattern_type!(u32 is 1..));

const BAD_FOO: Foo = Foo(Bar(unsafe { std::mem::transmute(0) }));
//~^ ERROR: constructing invalid value at .0.0: encountered 0

const CHAR_UNINIT: pattern_type!(char is 'A'..'Z') =
    //~^ ERROR: using uninitialized data, but this operation requires initialized memory
    unsafe { std::mem::transmute(std::mem::MaybeUninit::<u32>::uninit()) };

const CHAR_OOB_PAT: pattern_type!(char is 'A'..'Z') = unsafe { std::mem::transmute('a') };
//~^ ERROR: constructing invalid value: encountered 97, but expected something in the range 65..=89

const CHAR_OOB: pattern_type!(char is 'A'..'Z') = unsafe { std::mem::transmute(u32::MAX) };
//~^ ERROR: constructing invalid value: encountered 0xffffffff

fn main() {}
