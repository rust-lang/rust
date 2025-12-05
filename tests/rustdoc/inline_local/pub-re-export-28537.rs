// https://github.com/rust-lang/rust/issues/28537
#![crate_name="foo"]

#[doc(hidden)]
pub mod foo {
    pub struct Foo;
}

mod bar {
    pub use self::bar::Bar;
    mod bar {
        pub struct Bar;
    }
}

//@ has foo/struct.Foo.html
pub use foo::Foo;

//@ has foo/struct.Bar.html
pub use self::bar::Bar;
