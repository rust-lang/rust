struct Foo {
    foo: Vec<u32>,
}

impl Copy for Foo { } //~ ERROR cannot be implemented for this type

#[derive(Copy)]
struct Foo2<'a> { //~ ERROR cannot be implemented for this type
    ty: &'a mut bool,
}

enum EFoo {
    Bar { x: Vec<u32> },
    Baz,
}

impl Copy for EFoo { } //~ ERROR cannot be implemented for this type

#[derive(Copy)]
enum EFoo2<'a> { //~ ERROR cannot be implemented for this type
    Bar(&'a mut bool),
    Baz,
}

fn main() {
}
