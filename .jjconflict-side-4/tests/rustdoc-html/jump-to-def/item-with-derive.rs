// This test ensures that the item name will link to its item's page even if there
// is a `#[derive(...)]`.
// This is a regression test for <https://github.com/rust-lang/rust/issues/158050>.

//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/item-with-derive.rs.html'
//@ has - '//a[@href="../../foo/struct.Bar.html"]' 'Bar'
#[derive(Debug)]
pub struct Bar {
    x: u8,
}

// Same test with an enum just in case...
//@ has - '//a[@href="../../foo/enum.Blob.html"]' 'Blob'
#[derive(Debug)]
pub enum Blob {
    X,
}
