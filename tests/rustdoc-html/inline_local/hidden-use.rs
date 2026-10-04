mod private {
    pub struct Foo {}
}

//@ has hidden_use/index.html
//@ !hasraw - 'private'
//@ !hasraw - 'Foo'
//@ !has hidden_use/type.Foo.html
#[doc(hidden)]
pub use private::Foo;
