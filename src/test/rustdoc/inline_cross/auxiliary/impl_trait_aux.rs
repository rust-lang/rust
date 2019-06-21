pub fn func<'a>(_x: impl Clone + Into<Vec<u8>> + 'a) {}

pub struct Foo;

impl Foo {
    pub fn method<'a>(_x: impl Clone + Into<Vec<u8>> + 'a) {}
}
