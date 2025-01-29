#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

fn drop_pattern(arg: pattern_type!(u32 is 1..)) -> u32 {
    arg
}

fn drop_pattern_type_changing(arg: pattern_type!(u32 is 1..)) -> u64 {
    arg //~ ERROR mismatched types
}

fn drop_pattern_nested(arg: Option<pattern_type!(u32 is 1..)>) -> Option<u32> {
    arg //~ ERROR mismatched types
}

fn eq(a: pattern_type!(u32 is 1..), b: u32) -> bool {
    a == b
}

fn eq2(a: pattern_type!(u32 is 1..), b: pattern_type!(u32 is 1..)) -> bool {
    a == b
}

fn relax_pattern(arg: pattern_type!(u32 is 2..)) -> pattern_type!(u32 is 1..) {
    arg //~ ERROR mismatched types
}

#[rustfmt::skip]
fn arms(b: bool) -> u32 {
    if b {
        0
    } else {
        let x: pattern_type!(u32 is 1..) = 12;
        x
    }
}

#[rustfmt::skip]
fn arms2(b: bool) -> u32 {
    if b {
        let x: pattern_type!(u32 is 1..) = 12;
        x
    } else {
        0
    }
}

#[rustfmt::skip]
fn arms3(b: bool) -> pattern_type!(u32 is 1..) {
    if b {
        0 //~ ERROR mismatched types
    } else {
        let x: pattern_type!(u32 is 1..) = 12;
        x
    }
}

#[rustfmt::skip]
fn arms4(b: bool) -> pattern_type!(u32 is 1..) {
    if b {
        let x: pattern_type!(u32 is 1..) = 12;
        x
    } else {
        0 //~ ERROR mismatched types
    }
}

trait Foo {
    fn foo(&self) {}
}

impl Foo for u32 {}

fn foo() -> pattern_type!(u32 is 1..) {
    2
}

fn bar() {
    foo().foo();
    //~^ ERROR: no method named `foo` found for pattern type `(u32) is 1..`
}

fn main() {}
