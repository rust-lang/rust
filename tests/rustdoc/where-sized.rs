#![crate_name = "foo"]

// @has foo/fn.foo.html
// @has - '//*[@class="rust fn"]' 'pub fn foo<X, Y: ?Sized>(_: &X)'
// @has - '//*[@class="rust fn"]' 'where X: ?Sized,'
pub fn foo<X, Y: ?Sized>(_: &X) where X: ?Sized {}
