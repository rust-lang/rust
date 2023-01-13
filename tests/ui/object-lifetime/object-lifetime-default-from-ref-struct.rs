// run-pass
// Test that the lifetime of the enclosing `&` is used for the object
// lifetime bound.

// pretty-expanded FIXME #23616

#![allow(dead_code)]

use std::fmt::Display;

trait Test {
    fn foo(&self) { }
}

struct Ref<'a,T:'a+?Sized> {
    r: &'a T
}

struct Ref2<'a,'b,T:'a+'b+?Sized> {
    a: &'a T,
    b: &'b T
}

struct SomeStruct<'a> {
    t: Ref<'a, dyn Test>,
    u: Ref<'a, dyn Test+'a>,
}

fn a<'a>(t: Ref<'a, dyn Test>, mut ss: SomeStruct<'a>) {
    ss.t = t;
}

fn b<'a>(t: Ref<'a, dyn Test>, mut ss: SomeStruct<'a>) {
    ss.u = t;
}

fn c<'a>(t: Ref<'a, dyn Test+'a>, mut ss: SomeStruct<'a>) {
    ss.t = t;
}

fn d<'a>(t: Ref<'a, dyn Test+'a>, mut ss: SomeStruct<'a>) {
    ss.u = t;
}

fn e<'a>(_: Ref<'a, dyn Display+'static>) {}
fn g<'a, 'b>(_: Ref2<'a, 'b, dyn Display+'static>) {}


fn main() {
    // Inside a function body, we can just infer all
    // lifetimes, to allow Ref<'tmp, Display+'static>
    // and Ref2<'tmp, 'tmp, Display+'static>.
    let x = &0 as &(dyn Display+'static);
    let r: Ref<dyn Display> = Ref { r: x };
    let r2: Ref2<dyn Display> = Ref2 { a: x, b: x };
    e(r);
    g(r2);
}
