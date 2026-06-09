// https://github.com/rust-lang/rust/issues/35488
#![crate_name="foo"]

mod foo {
    pub enum Foo {
        Bar,
    }
    pub use self::Foo::*;
}

//@ has 'foo/index.html' '//code' 'pub use self::Foo::*;'
//@ has 'foo/enum.Foo.html'
pub use self::foo::*;

//@ has 'foo/index.html' '//code' 'pub use std::option::Option::None;'
pub use std::option::Option::None;
