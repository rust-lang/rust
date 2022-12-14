#![crate_name = "foo"]
#![feature(rustdoc_internals)]

// @matches 'foo/index.html' '//h1' 'Crate foo'
// @matches 'foo/index.html' '//h2[@class="location"]' 'Crate foo'

// @matches 'foo/foo_mod/index.html' '//h1' 'Module foo::foo_mod'
// @matches 'foo/foo_mod/index.html' '//h2[@class="location"]' 'Module foo_mod'
pub mod foo_mod {
    pub struct __Thing {}
}

extern "C" {
    // @matches 'foo/fn.foo_ffn.html' '//h1' 'Function foo::foo_ffn'
    pub fn foo_ffn();
}

// @matches 'foo/fn.foo_fn.html' '//h1' 'Function foo::foo_fn'
pub fn foo_fn() {}

// @matches 'foo/trait.FooTrait.html' '//h1' 'Trait foo::FooTrait'
// @matches 'foo/trait.FooTrait.html' '//h2[@class="location"]' 'FooTrait'
pub trait FooTrait {}

// @matches 'foo/struct.FooStruct.html' '//h1' 'Struct foo::FooStruct'
// @matches 'foo/struct.FooStruct.html' '//h2[@class="location"]' 'FooStruct'
pub struct FooStruct;

// @matches 'foo/enum.FooEnum.html' '//h1' 'Enum foo::FooEnum'
// @matches 'foo/enum.FooEnum.html' '//h2[@class="location"]' 'FooEnum'
pub enum FooEnum {}

// @matches 'foo/type.FooType.html' '//h1' 'Type Definition foo::FooType'
// @matches 'foo/type.FooType.html' '//h2[@class="location"]' 'FooType'
pub type FooType = FooStruct;

// @matches 'foo/macro.foo_macro.html' '//h1' 'Macro foo::foo_macro'
#[macro_export]
macro_rules! foo_macro {
    () => {};
}

// @matches 'foo/primitive.bool.html' '//h1' 'Primitive Type bool'
#[doc(primitive = "bool")]
mod bool {}

// @matches 'foo/static.FOO_STATIC.html' '//h1' 'Static foo::FOO_STATIC'
pub static FOO_STATIC: FooStruct = FooStruct;

extern "C" {
    // @matches 'foo/static.FOO_FSTATIC.html' '//h1' 'Static foo::FOO_FSTATIC'
    pub static FOO_FSTATIC: FooStruct;
}

// @matches 'foo/constant.FOO_CONSTANT.html' '//h1' 'Constant foo::FOO_CONSTANT'
pub const FOO_CONSTANT: FooStruct = FooStruct;
