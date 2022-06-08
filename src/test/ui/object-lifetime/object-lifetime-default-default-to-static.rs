// run-pass
// Test that `Box<Test>` is equivalent to `Box<Test+'static>`, both in
// fields and fn arguments.

// pretty-expanded FIXME #23616

#![allow(dead_code)]

trait Test {
    fn foo(&self) { }
}

struct SomeStruct {
    t: Box<dyn Test>,
    u: Box<dyn Test+'static>,
}

fn a(t: Box<dyn Test>, mut ss: SomeStruct) {
    ss.t = t;
}

fn b(t: Box<dyn Test+'static>, mut ss: SomeStruct) {
    ss.t = t;
}

fn c(t: Box<dyn Test>, mut ss: SomeStruct) {
    ss.u = t;
}

fn d(t: Box<dyn Test+'static>, mut ss: SomeStruct) {
    ss.u = t;
}

// Check that closures obey the same rules.
fn e() {
    let _ = |t: Box<dyn Test>, mut ss: SomeStruct| {
        ss.t = t;
    };
    let _ = |t: Box<dyn Test+'static>, mut ss: SomeStruct| {
        ss.t = t;
    };
    let _ = |t: Box<dyn Test>, mut ss: SomeStruct| {
        ss.u = t;
    };
    let _ = |t: Box<dyn Test+'static>, mut ss: SomeStruct| {
        ss.u = t;
    };
}

// Check that bare fns and Fn trait obey the same rules.
fn f() {
    let _: fn(Box<dyn Test>, SomeStruct) = a;
    let _: fn(Box<dyn Test + 'static>, SomeStruct) = a;
    let _: &dyn Fn(Box<dyn Test>, SomeStruct) = &a;
    let _: &dyn Fn(Box<dyn Test + 'static>, SomeStruct) = &a;
}

// But closure return type does not need to.
fn g<'a>(t: Box<dyn Test + 'a>) {
    let _ = || -> Box<dyn Test> { t };
}

fn main() {
}
