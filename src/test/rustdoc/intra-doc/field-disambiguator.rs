#![crate_name = "foo"]

pub struct Foo {
    pub bar: i32,
}

impl Foo {
    /// [`Foo::bar()`] gets a reference to [`field@Foo::bar`].
    /// What about [without disambiguators](Foo::bar)?
    // @has foo/struct.Foo.html '//a[@href="../foo/struct.Foo.html#method.bar"]' 'Foo::bar()'
    // @has foo/struct.Foo.html '//a[@href="../foo/struct.Foo.html#structfield.bar"]' 'Foo::bar'
    // @!has foo/struct.Foo.html '//a[@href="../foo/struct.Foo.html#structfield.bar"]' 'field'
    // @has foo/struct.Foo.html '//a[@href="../foo/struct.Foo.html#method.bar"]' 'without disambiguators'
    pub fn bar(&self) -> i32 { self.bar }
}
