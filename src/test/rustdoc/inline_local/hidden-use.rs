mod private {
    pub struct Foo {}
}

// @has hidden_use/index.html
// @!has - 'private'
// @!has - 'Foo'
// @!has hidden_use/struct.Foo.html
#[doc(hidden)]
pub use private::Foo;
