#![crate_name = "foo"]

// @has foo/struct.Bar.html
// @!has - '//div[@id="impl-Sized"]'
pub struct Bar {
    a: u16,
}

// @has foo/struct.Foo.html
// @!has - '//div[@id="impl-Sized"]'
pub struct Foo<T: ?Sized>(T);

// @has foo/struct.Unsized.html
// @has - '//div[@id="impl-Sized"]/code' 'impl !Sized for Unsized'
pub struct Unsized {
    data: [u8],
}
