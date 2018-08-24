#![feature(extern_types)]

pub mod foo_mod {}
extern "C" {
    pub fn foo_ffn();
    pub static FOO_FSTATIC: FooStruct;
    pub type FooFType;
}
pub fn foo_fn() {}
pub trait FooTrait {}
pub struct FooStruct;
pub enum FooEnum {}
pub union FooUnion {
    x: (),
}
pub type FooType = FooStruct;
pub static FOO_STATIC: FooStruct = FooStruct;
pub const FOO_CONSTANT: FooStruct = FooStruct;
#[macro_export]
macro_rules! foo_macro {
    () => ();
}
