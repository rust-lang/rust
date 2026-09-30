pub struct Foo;

// just so that `Foo` doesn't show up on `Bar`s sidebar
pub mod bar {
    pub trait Bar {}
}

impl Foo {
    pub fn new() -> Foo { Foo }
}

impl bar::Bar for Foo {}
