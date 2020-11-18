#![crate_name = "quix"]
pub trait Foo {
    // @has quix/trait.Foo.html '//a[@href="../src/quix/trait-src-link.rs.html#4"]' '[src]'
    fn required();

    // @has quix/trait.Foo.html '//a[@href="../src/quix/trait-src-link.rs.html#7"]' '[src]'
    fn provided() {}
}

pub struct Bar;

impl Foo for Bar {
    // @has quix/struct.Bar.html '//a[@href="../src/quix/trait-src-link.rs.html#14"]' '[src]'
    fn required() {}
    // @has quix/struct.Bar.html '//a[@href="../src/quix/trait-src-link.rs.html#7"]' '[src]'
}

pub struct Baz;

impl Foo for Baz {
    // @has quix/struct.Baz.html '//a[@href="../src/quix/trait-src-link.rs.html#22"]' '[src]'
    fn required() {}

    // @has quix/struct.Baz.html '//a[@href="../src/quix/trait-src-link.rs.html#25"]' '[src]'
    fn provided() {}
}
