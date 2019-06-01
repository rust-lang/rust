#![crate_name = "foo"]

// ignore-tidy-linelength

pub trait Foo {
    // @has foo/trait.Foo.html '//h3[@id="tymethod.foo"]//span[@class="docblock attributes"]' '#[must_use]'
    #[must_use]
    fn foo();
}

#[must_use]
pub struct Bar;

impl Bar {
    // @has foo/struct.Bar.html '//h4[@id="method.bar"]//span[@class="docblock attributes"]' '#[must_use]'
    #[must_use]
    pub fn bar() {}

    // @has foo/struct.Bar.html '//h4[@id="method.bar2"]//span[@class="docblock attributes"]' '#[must_use]'
    #[must_use]
    pub fn bar2() {}
}
