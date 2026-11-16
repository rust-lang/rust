#![crate_name = "quix"]
pub trait Foo {
    //@ has quix/trait.Foo.html '//a[@href="../src/quix/trait-src-link.rs.html#4"]' 'Source'
    fn required();

    //@ has quix/trait.Foo.html '//a[@href="../src/quix/trait-src-link.rs.html#7"]' 'Source'
    fn provided() {}
}

pub struct Bar;

impl Foo for Bar {
    //@ has quix/struct.Bar.html '//a[@href="../src/quix/trait-src-link.rs.html#14"]' 'Source'
    fn required() {}
    // For the `provided` method.
    //@ has quix/struct.Bar.html '//a[@href="../src/quix/trait-src-link.rs.html#7"]' 'Source'
}

pub struct Baz;

impl Foo for Baz {
    //@ has quix/struct.Baz.html '//a[@href="../src/quix/trait-src-link.rs.html#23"]' 'Source'
    fn required() {}

    //@ has quix/struct.Baz.html '//a[@href="../src/quix/trait-src-link.rs.html#26"]' 'Source'
    fn provided() {}
}
