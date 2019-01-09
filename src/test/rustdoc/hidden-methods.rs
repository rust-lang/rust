#![crate_name = "foo"]

#[doc(hidden)]
pub mod hidden {
    pub struct Foo;

    impl Foo {
        #[doc(hidden)]
        pub fn this_should_be_hidden() {}
    }

    pub struct Bar;

    impl Bar {
        fn this_should_be_hidden() {}
    }
}

// @has foo/struct.Foo.html
// @!has - 'Methods'
// @!has - '//code' 'impl Foo'
// @!has - 'this_should_be_hidden'
pub use hidden::Foo;

// @has foo/struct.Bar.html
// @!has - 'Methods'
// @!has - '//code' 'impl Bar'
// @!has - 'this_should_be_hidden'
pub use hidden::Bar;
