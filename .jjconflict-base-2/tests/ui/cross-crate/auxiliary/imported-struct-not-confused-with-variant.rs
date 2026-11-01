//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/19293
pub struct Foo (pub isize);
pub enum MyEnum {
    Foo(Foo),
}
