// ignore-compare-mode-nll
// revisions: base nll
// [nll]compile-flags: -Zborrowck=mir

// Test that the lifetime from the enclosing `&` is "inherited"
// through the `Box` struct.

#![allow(dead_code)]

trait Test {
    fn foo(&self) { }
}

struct SomeStruct<'a> {
    t: &'a Box<dyn Test>,
}

fn c<'a>(t: &'a Box<dyn Test+'a>, mut ss: SomeStruct<'a>) {
    ss.t = t;
    //[base]~^ ERROR mismatched types
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {
}
