#![crate_name = "foo"]

// @has foo/trait.Foo.html
// @has - '//details[@class="rustdoc-toggle"]//code' 'bar'
pub trait Foo {
    fn bar() -> ();
}
