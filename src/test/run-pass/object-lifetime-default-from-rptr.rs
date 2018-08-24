// Test that the lifetime of the enclosing `&` is used for the object
// lifetime bound.

// pretty-expanded FIXME #23616

#![allow(dead_code)]

use std::fmt::Display;

trait Test {
    fn foo(&self) { }
}

struct SomeStruct<'a> {
    t: &'a Test,
    u: &'a (Test+'a),
}

fn a<'a>(t: &'a Test, mut ss: SomeStruct<'a>) {
    ss.t = t;
}

fn b<'a>(t: &'a Test, mut ss: SomeStruct<'a>) {
    ss.u = t;
}

fn c<'a>(t: &'a (Test+'a), mut ss: SomeStruct<'a>) {
    ss.t = t;
}

fn d<'a>(t: &'a (Test+'a), mut ss: SomeStruct<'a>) {
    ss.u = t;
}

fn e<'a>(_: &'a (Display+'static)) {}

fn main() {
    // Inside a function body, we can just infer both
    // lifetimes, to allow &'tmp (Display+'static).
    e(&0 as &Display);
}
