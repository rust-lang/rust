// ignore-cross-compile

#![crate_name = "foo"]

pub trait SomeTrait {}
pub struct SomeStruct;

// @has foo/trait.SomeTrait.html '//a/@href' '../src/foo/issue-43893.rs.html#9'
impl SomeTrait for usize {}

// @has foo/trait.SomeTrait.html '//a/@href' '../src/foo/issue-43893.rs.html#12-14'
impl SomeTrait for SomeStruct {
    // deliberately multi-line impl
}

pub trait AnotherTrait {}

// @has foo/trait.AnotherTrait.html '//a/@href' '../src/foo/issue-43893.rs.html#19'
impl<T> AnotherTrait for T {}
