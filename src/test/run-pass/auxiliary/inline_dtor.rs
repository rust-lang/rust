#![crate_name="inline_dtor"]

pub struct Foo;

impl Drop for Foo {
    #[inline]
    fn drop(&mut self) {}
}
