// ignore-cross-compile

#![crate_name = "foo"]

pub trait SomeTrait {}
pub struct SomeStruct;

// @has foo/trait.SomeTrait.html '//a/@href' '../src/foo/issue-43893.rs.html#19'
impl SomeTrait for usize {}

// @has foo/trait.SomeTrait.html '//a/@href' '../src/foo/issue-43893.rs.html#22-24'
impl SomeTrait for SomeStruct {
    // deliberately multi-line impl
}

pub trait AnotherTrait {}

// @has foo/trait.AnotherTrait.html '//a/@href' '../src/foo/issue-43893.rs.html#29'
impl<T> AnotherTrait for T {}
