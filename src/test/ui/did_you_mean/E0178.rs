#![allow(bare_trait_objects)]

trait Foo {}

struct Bar<'a> {
    w: &'a Foo + Copy, //~ ERROR expected a path
    x: &'a Foo + 'a, //~ ERROR expected a path
    y: &'a mut Foo + 'a, //~ ERROR expected a path
    z: fn() -> Foo + 'a, //~ ERROR expected a path
}

fn main() {
}
