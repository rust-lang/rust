pub trait Foo {
    const FOO: usize;
}

pub struct Bar;

impl Bar {
    pub const BAR: usize = 3;
}
