#![crate_name = "foo"]

// @has foo/fn.foo.html
// @has - '//*[@class="rust fn"]' 'pub fn foo<X, Y>(_: &X)'
// @has - '//*[@class="rust fn"]' 'where Y: ?Sized,'
// @has - '//*[@class="rust fn"]' 'X: ?Sized,'
pub fn foo<X, Y: ?Sized>(_: &X)
where
    X: ?Sized,
{
}
