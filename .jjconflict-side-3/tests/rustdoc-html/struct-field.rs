#![crate_name = "foo"]


//@ has foo/index.html '//*[@class="docblock"]/p/a[@href="struct.Foo.html#structfield.bar"]' 'Foo::bar'
//@ has foo/index.html '//*[@class="docblock"]/p/a[@href="union.Bar.html#structfield.foo"]' 'Bar::foo'
//@ has foo/index.html '//*[@class="docblock"]/p/a[@href="enum.Uniooon.html#variant.X"]' 'Uniooon::X'

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
