// https://github.com/rust-lang/rust/issues/53689
//@ aux-build:issue-53689.rs

#![crate_name = "foo"]

extern crate issue_53689;

//@ has foo/trait.MyTrait.html
//@ !hasraw - 'MyStruct'
//@ count - '//*[h3="impl<T> MyTrait for T"]' 1
pub trait MyTrait {}

impl<T> MyTrait for T {}

mod a {
    pub use issue_53689::MyStruct;
}
