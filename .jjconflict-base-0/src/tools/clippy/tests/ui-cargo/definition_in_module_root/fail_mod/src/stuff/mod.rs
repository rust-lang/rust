#![warn(clippy::definition_in_module_root)]

// Every definition kind should trigger the lint.
pub struct Foo;
pub enum Bar {
    A,
    B,
}
pub fn baz() {}
pub const QUUX: u32 = 1;
pub trait Trait1 {}
pub type Alias<T> = std::result::Result<T, String>;
impl Foo {
    pub fn method(&self) {}
}

// macro_export macros are allowed.
#[macro_export]
macro_rules! ok {
    () => {};
}
