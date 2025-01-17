//@ run-pass
// Test that the lifetime from the enclosing `&` is "inherited"
// through the `Box` struct.


#![allow(dead_code)]

trait Test {
    fn foo(&self) { }
}

struct SomeStruct<'a> {
    t: &'a Box<dyn Test>,
    u: &'a Box<dyn Test+'a>,
}

fn a<'a>(t: &'a Box<dyn Test>, mut ss: SomeStruct<'a>) {
    ss.t = t;
}

fn b<'a>(t: &'a Box<dyn Test>, mut ss: SomeStruct<'a>) {
    ss.u = t;
}

// see also ui/object-lifetime/object-lifetime-default-from-rptr-box-error.rs

fn d<'a>(t: &'a Box<dyn Test+'a>, mut ss: SomeStruct<'a>) {
    ss.u = t;
}

fn main() {
}
