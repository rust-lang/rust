#![warn(clippy::disallowed_method)]
#![allow(clippy::no_effect, clippy::many_single_char_names)]

struct ImplStruct;

trait Baz {
    fn bad_method(self);
}

impl Baz for ImplStruct {
    fn bad_method(self) {}
}

struct Foo;

impl Foo {
    fn bad_method(self) {}
}

struct StaticStruct;

trait Quux {
    fn bad_method();
}

impl Quux for StaticStruct {
    fn bad_method() {}
}

struct NormalStruct;

impl NormalStruct {
    fn bad_method(self) {}
}

struct AttrStruct {
    bad_method: i32,
}

fn main() {
    let b = ImplStruct;
    let f = Foo;
    let c = ImplStruct;
    let n = NormalStruct;
    let a = AttrStruct{ bad_method: 5 };

    // lint these
    b.bad_method();
    c.bad_method();
    f.bad_method();
    // these are good
    // good because not a method call (ExprKind => Call)
    StaticStruct::bad_method();
    n.bad_method();
    a.bad_method;
}
