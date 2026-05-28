use std::rc::Rc;

struct Foo<'a> { //~ ERROR recursive types `Foo` and `Bar` have infinite size
    bar: Bar<'a>,
    b: Rc<Bar<'a>>,
}

struct Bar<'a> {
    y: (Foo<'a>, Foo<'a>),
    z: Option<Bar<'a>>,
    a: &'a Foo<'a>,
    c: &'a [Bar<'a>],
    d: [Bar<'a>; 1],
    e: Foo<'a>,
    x: Bar<'a>,
}

fn main() {}
