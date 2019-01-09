#![crate_name = "foo"]

// @has foo/fn.bar.html
// @has - '//*[@class="rust fn"]' 'pub const fn bar() -> '
/// foo
pub const fn bar() -> usize {
    2
}

// @has foo/struct.Foo.html
// @has - '//*[@class="method"]' 'const fn new()'
pub struct Foo(usize);

impl Foo {
    pub const fn new() -> Foo { Foo(0) }
}
