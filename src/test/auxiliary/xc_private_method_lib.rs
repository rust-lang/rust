#[crate_type="lib"];

pub struct Foo {
    x: int
}

impl Foo {
    fn new() -> Foo { Foo { x: 1 } }
}
