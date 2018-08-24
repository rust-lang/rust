struct Foo {
    foo: Vec<u32>,
}

impl Copy for Foo { } //~ ERROR may not be implemented for this type

#[derive(Copy)] //~ ERROR may not be implemented for this type
struct Foo2<'a> {
    ty: &'a mut bool,
}

enum EFoo {
    Bar { x: Vec<u32> },
    Baz,
}

impl Copy for EFoo { } //~ ERROR may not be implemented for this type

#[derive(Copy)] //~ ERROR may not be implemented for this type
enum EFoo2<'a> {
    Bar(&'a mut bool),
    Baz,
}

fn main() {
}
