struct Foo {
    x: isize
}

impl Fo { //~ ERROR cannot find type `Fo`
    fn foo() {}
}

fn main() {}
