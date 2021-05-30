#![crate_name = "foo"]

// @has foo/trait.Foo.html
// @!has - '//details[@class="rustdoc-toggle"]//code' 'bar'
// @has - '//code' 'bar'
// @has - '//details[@class="rustdoc-toggle"]//code' 'foo'
pub trait Foo {
    fn bar() -> ();
    /// hello
    fn foo();
}
