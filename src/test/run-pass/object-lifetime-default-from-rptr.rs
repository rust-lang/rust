// Test that the lifetime of the enclosing `&` is used for the object
// lifetime bound.

// pretty-expanded FIXME #23616

#![allow(dead_code)]

use std::fmt::Display;

trait Test {
    fn foo(&self) { }
}

struct SomeStruct<'a> {
    t: &'a dyn Test,
    u: &'a (dyn Test+'a),
}

fn a<'a>(t: &'a dyn Test, mut ss: SomeStruct<'a>) {
    ss.t = t;
}

fn b<'a>(t: &'a dyn Test, mut ss: SomeStruct<'a>) {
    ss.u = t;
}

fn c<'a>(t: &'a (dyn Test+'a), mut ss: SomeStruct<'a>) {
    ss.t = t;
}

fn d<'a>(t: &'a (dyn Test+'a), mut ss: SomeStruct<'a>) {
    ss.u = t;
}

fn e<'a>(_: &'a (dyn Display+'static)) {}

fn main() {
    // Inside a function body, we can just infer both
    // lifetimes, to allow &'tmp (Display+'static).
    e(&0 as &dyn Display);
}
