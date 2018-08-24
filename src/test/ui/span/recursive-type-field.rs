use std::rc::Rc;

struct Foo<'a> { //~ ERROR recursive type
    bar: Bar<'a>,
    b: Rc<Bar<'a>>,
}

struct Bar<'a> { //~ ERROR recursive type
    y: (Foo<'a>, Foo<'a>),
    z: Option<Bar<'a>>,
    a: &'a Foo<'a>,
    c: &'a [Bar<'a>],
    d: [Bar<'a>; 1],
    e: Foo<'a>,
    x: Bar<'a>,
}

fn main() {}
