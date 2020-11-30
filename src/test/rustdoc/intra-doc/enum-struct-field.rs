#![crate_name = "foo"]

pub enum Foo {
    X {
        y: u8,
    }
}

/// Hello
///
/// I want [Foo::X::y].
pub fn foo() {}

// @has foo/fn.foo.html '//a/@href' '../foo/enum.Foo.html#variant.X.field.y'
