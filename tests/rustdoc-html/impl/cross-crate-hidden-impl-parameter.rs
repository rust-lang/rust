// Issue #86448: test for cross-crate `doc(hidden)`
#![crate_name = "foo"]

//@ aux-build:cross-crate-hidden-impl-parameter.rs
extern crate cross_crate_hidden_impl_parameter;

pub use ::cross_crate_hidden_impl_parameter::{HiddenType, HiddenTrait}; // OK, not re-exported

pub enum MyLibType {}

//@ !has foo/enum.MyLibType.html '//*[@id="impl-From%3CHiddenType%3E"]' 'impl From<HiddenType> for MyLibType'
impl From<HiddenType> for MyLibType {
    fn from(it: HiddenType) -> MyLibType {
        match it {}
    }
}

pub struct T<T>(T);

//@ !has foo/enum.MyLibType.html '//*[@id="impl-From%3CT%3CT%3CT%3CT%3CHiddenType%3E%3E%3E%3E%3E"]' 'impl From<T<T<T<T<HiddenType>>>>> for MyLibType'
impl From<T<T<T<T<HiddenType>>>>> for MyLibType {
    fn from(it: T<T<T<T<HiddenType>>>>) -> MyLibType {
        todo!()
    }
}

//@ !has foo/enum.MyLibType.html '//*[@id="impl-HiddenTrait"]' 'impl HiddenTrait for MyLibType'
impl HiddenTrait for MyLibType {}

//@ !has foo/struct.T.html '//*[@id="impl-From%3CMyLibType%3E"]' 'impl From<MyLibType> for T<T<T<T<HiddenType>>>>'
impl From<MyLibType> for T<T<T<T<HiddenType>>>> {
    fn from(it: MyLibType) -> T<T<T<T<HiddenType>>>> {
        match it {}
    }
}
