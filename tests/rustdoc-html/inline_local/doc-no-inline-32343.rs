// https://github.com/rust-lang/rust/issues/32343
#![crate_name="foobar"]

//@ !has foobar/struct.Foo.html
//@ has foobar/index.html
//@ has - '//code' 'pub use foo::Foo'
//@ !has - '//code/a' 'Foo'
#[doc(no_inline)]
pub use foo::Foo;

//@ !has foobar/struct.Bar.html
//@ has foobar/index.html
//@ has - '//code' 'pub use foo::Bar'
//@ has - '//code/a' 'Bar'
#[doc(no_inline)]
pub use foo::Bar;

mod foo {
    pub struct Foo;
    pub struct Bar;
}

pub mod bar {
    //@ has foobar/bar/struct.Bar.html
    pub use ::foo::Bar;
}
