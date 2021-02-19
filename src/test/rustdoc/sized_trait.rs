#![crate_name = "foo"]

// @has foo/struct.Bar.html
// @!has - '//h3[@id="impl-Sized"]'
pub struct Bar {
    a: u16,
}

// @has foo/struct.Foo.html
// @!has - '//h3[@id="impl-Sized"]'
pub struct Foo<T: ?Sized>(T);

// @has foo/struct.Unsized.html
// @has - '//h3[@id="impl-Sized"]/code' 'impl !Sized for Unsized'
pub struct Unsized {
    data: [u8],
}
