//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/43893

#![crate_name = "foo"]

pub trait SomeTrait {}
pub struct SomeStruct;

//@ has foo/trait.SomeTrait.html '//a/@href' '../src/foo/src-links-implementor-43893.rs.html#11'
impl SomeTrait for usize {}

//@ has foo/trait.SomeTrait.html '//a/@href' '../src/foo/src-links-implementor-43893.rs.html#14-16'
impl SomeTrait for SomeStruct {
    // deliberately multi-line impl
}

pub trait AnotherTrait {}

//@ has foo/trait.AnotherTrait.html '//a/@href' '../src/foo/src-links-implementor-43893.rs.html#21'
impl<T> AnotherTrait for T {}
