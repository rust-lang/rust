struct Foo {
    x: isize
}

impl Fo { //~ ERROR cannot find type `Fo` in this scope
    fn foo() {}
}

fn main() {}
