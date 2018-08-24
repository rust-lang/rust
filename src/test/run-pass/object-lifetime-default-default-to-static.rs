// Test that `Box<Test>` is equivalent to `Box<Test+'static>`, both in
// fields and fn arguments.

// pretty-expanded FIXME #23616

#![allow(dead_code)]

trait Test {
    fn foo(&self) { }
}

struct SomeStruct {
    t: Box<Test>,
    u: Box<Test+'static>,
}

fn a(t: Box<Test>, mut ss: SomeStruct) {
    ss.t = t;
}

fn b(t: Box<Test+'static>, mut ss: SomeStruct) {
    ss.t = t;
}

fn c(t: Box<Test>, mut ss: SomeStruct) {
    ss.u = t;
}

fn d(t: Box<Test+'static>, mut ss: SomeStruct) {
    ss.u = t;
}

fn main() {
}
