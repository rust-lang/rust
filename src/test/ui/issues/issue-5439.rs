#![feature(box_syntax)]

struct Foo {
    foo: isize,
}

struct Bar {
    bar: isize,
}

impl Bar {
    fn make_foo (&self, i: isize) -> Box<Foo> {
        return box Foo { nonexistent: self, foo: i }; //~ ERROR: no field named
    }
}

fn main () {
    let bar = Bar { bar: 1 };
    let foo = bar.make_foo(2);
    println!("{}", foo.foo);
}
