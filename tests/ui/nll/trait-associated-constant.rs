// Test cases where we put various lifetime constraints on trait
// associated constants.

#![feature(rustc_attrs)]

use std::option::Option;

trait Anything<'a: 'b, 'b> {
    const AC: Option<&'b str>;
}

struct OKStruct1 { }

impl<'a: 'b, 'b> Anything<'a, 'b> for OKStruct1 {
    const AC: Option<&'b str> = None;
}

struct FailStruct { }

impl<'a: 'b, 'b, 'c> Anything<'a, 'b> for FailStruct {
    const AC: Option<&'c str> = None;
    //~^ ERROR: const not compatible with trait
}

struct OKStruct2 { }

impl<'a: 'b, 'b> Anything<'a, 'b> for OKStruct2 {
    const AC: Option<&'a str> = None;
}

fn main() {}
