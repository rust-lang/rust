// compile-flags: --document-private-items

#![crate_name = "foo"]

// @has 'foo/fn.foo.html' '//pre' 'fn foo'
// !@has 'foo/fn.foo.html' '//pre' 'pub'
fn foo() {}

mod bar {
    // @has 'foo/bar/fn.baz.html' '//pre' 'fn baz'
    // !@has 'foo/bar/fn.baz.html' '//pre' 'pub'
    fn baz() {}
}
