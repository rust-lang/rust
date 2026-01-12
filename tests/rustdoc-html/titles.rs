#![crate_name = "foo"]
#![feature(rustc_attrs)]

//@ matches 'foo/index.html' '//h1' 'Crate foo'
//@ matches 'foo/index.html' '//div[@class="sidebar-crate"]/h2/a' 'foo'
//@ count 'foo/index.html' '//h2[@class="location"]' 0

//@ matches 'foo/foo_mod/index.html' '//h1' 'Module foo_mod'
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo'
//@ matches - '//h2[@class="location"]' 'Module foo_mod'
pub mod foo_mod {
    pub struct __Thing {}
}

extern "C" {
    //@ matches 'foo/fn.foo_ffn.html' '//h1' 'Function foo_ffn'
    //@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo'
    pub fn foo_ffn();
}

//@ matches 'foo/fn.foo_fn.html' '//h1' 'Function foo_fn'
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo'
pub fn foo_fn() {}

//@ matches 'foo/trait.FooTrait.html' '//h1' 'Trait FooTrait'
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo'
//@ matches - '//h2[@class="location"]' 'FooTrait'
pub trait FooTrait {}

//@ matches 'foo/struct.FooStruct.html' '//h1' 'Struct FooStruct'
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo'
//@ matches - '//h2[@class="location"]' 'FooStruct'
pub struct FooStruct;

//@ matches 'foo/enum.FooEnum.html' '//h1' 'Enum FooEnum'
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo'
//@ matches - '//h2[@class="location"]' 'FooEnum'
pub enum FooEnum {}

//@ matches 'foo/type.FooType.html' '//h1' 'Type Alias FooType'
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo'
//@ matches - '//h2[@class="location"]' 'FooType'
pub type FooType = FooStruct;

//@ matches 'foo/macro.foo_macro.html' '//h1' 'Macro foo_macro'
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo'
#[macro_export]
macro_rules! foo_macro {
    () => {};
}

//@ matches 'foo/primitive.bool.html' '//h1' 'Primitive Type bool'
//@ count - '//*[@class="rustdoc-breadcrumbs"]' 0
#[rustc_doc_primitive = "bool"]
mod bool {}

//@ matches 'foo/static.FOO_STATIC.html' '//h1' 'Static FOO_STATIC'
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo'
pub static FOO_STATIC: FooStruct = FooStruct;

extern "C" {
    //@ matches 'foo/static.FOO_FSTATIC.html' '//h1' 'Static FOO_FSTATIC'
    //@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo'
    pub static FOO_FSTATIC: FooStruct;
}

//@ matches 'foo/constant.FOO_CONSTANT.html' '//h1' 'Constant FOO_CONSTANT'
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo'
pub const FOO_CONSTANT: FooStruct = FooStruct;
