//@ run-pass
#![allow(dead_code)]
//@ aux-build:issue-10028.rs


extern crate issue_10028 as issue10028;

use issue10028::ZeroLengthThingWithDestructor;

struct Foo {
    zero_length_thing: ZeroLengthThingWithDestructor
}

fn make_foo() -> Foo {
    Foo { zero_length_thing: ZeroLengthThingWithDestructor::new() }
}

fn main() {
    let _f:Foo = make_foo();
}
