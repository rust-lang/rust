// aux-build:issue-53689.rs

#![crate_name = "foo"]

extern crate issue_53689;

// @has foo/trait.MyTrait.html
// @!has - 'MyStruct'
// @count - '//*[code="impl<T> MyTrait for T"]' 1
pub trait MyTrait {}

impl<T> MyTrait for T {}

mod a {
    pub use issue_53689::MyStruct;
}
