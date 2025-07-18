//@ run-pass
// Test that even with prior inferred parameters, object lifetimes of objects after are still
// valid.


#![allow(dead_code)]

trait Test {
    fn foo(&self) { }
}

struct Foo;
impl Test for Foo {}

struct SomeStruct<'a> {
    t: &'a dyn Test,
    u: &'a (dyn Test+'a),
}

fn a<'a, const N: usize>(_: [u8; N], t: &'a (dyn Test+'a), mut ss: SomeStruct<'a>) {
    ss.t = t;
}

fn b<'a, T>(_: T, t: &'a (dyn Test+'a), mut ss: SomeStruct<'a>) {
    ss.u = t;
}

fn main() {
    // Inside a function body, we can just infer both
    // lifetimes, to allow &'tmp (Display+'static).
    a::<_>([], &Foo as &dyn Test, SomeStruct{t:&Foo,u:&Foo});
    b::<_>(0u8, &Foo as &dyn Test, SomeStruct{t:&Foo,u:&Foo});
}
