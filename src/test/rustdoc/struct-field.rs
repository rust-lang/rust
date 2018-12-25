#![crate_name = "foo"]

// ignore-tidy-linelength

// @has foo/index.html '//*[@class="docblock"]/p/a[@href="../foo/struct.Foo.html#structfield.bar"]' 'Foo::bar'
// @has foo/index.html '//*[@class="docblock"]/p/a[@href="../foo/union.Bar.html#structfield.foo"]' 'Bar::foo'
// @has foo/index.html '//*[@class="docblock"]/p/a[@href="../foo/enum.Uniooon.html#X.v"]' 'Uniooon::X'

//! Test with [Foo::bar], [Bar::foo], [Uniooon::X]

pub struct Foo {
    pub bar: usize,
}

pub union Bar {
    pub foo: u32,
}

pub enum Uniooon {
    F,
    X,
    Y,
}
