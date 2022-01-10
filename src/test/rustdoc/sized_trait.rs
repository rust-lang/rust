#![crate_name = "foo"]

// @has foo/struct.Bar.html
// @!has - '//*[@id="impl-Sized"]'
pub struct Bar {
    a: u16,
}

// @has foo/struct.Foo.html
// @!has - '//*[@id="impl-Sized"]'
pub struct Foo<T: ?Sized>(T);

// @has foo/struct.Unsized.html
// @has - '//div[@id="impl-Sized-for-Unsized"]//h3[@class="code-header in-band"]' 'impl !Sized for Unsized'
>>>>>>> 083cf2a97a8... rustdoc: Add more semantic information to impl ids
pub struct Unsized {
    data: [u8],
}
