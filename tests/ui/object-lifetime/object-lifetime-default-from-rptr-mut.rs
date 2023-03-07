// run-pass
// Test that the lifetime of the enclosing `&` is used for the object
// lifetime bound.

// pretty-expanded FIXME #23616

#![allow(dead_code)]

trait Test {
    fn foo(&self) { }
}

struct SomeStruct<'a> {
    t: &'a mut dyn Test,
    u: &'a mut (dyn Test+'a),
}

fn a<'a>(t: &'a mut dyn Test, mut ss: SomeStruct<'a>) {
    ss.t = t;
}

fn b<'a>(t: &'a mut dyn Test, mut ss: SomeStruct<'a>) {
    ss.u = t;
}

fn c<'a>(t: &'a mut (dyn Test+'a), mut ss: SomeStruct<'a>) {
    ss.t = t;
}

fn d<'a>(t: &'a mut (dyn Test+'a), mut ss: SomeStruct<'a>) {
    ss.u = t;
}


fn main() {
}
