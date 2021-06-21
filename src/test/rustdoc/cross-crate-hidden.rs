// Issue #86448: test for cross-crate `doc(hidden)`
#![crate_name = "foo"]

// aux-build:cross-crate-hidden.rs
extern crate cross_crate_hidden;

pub use ::cross_crate_hidden::HiddenType; // OK, not re-exported

pub enum MyLibType {}

// @!has foo/enum.MyLibType.html '//*[@id="impl-From%3CHiddenType%3E"]' 'impl From<HiddenType> for MyLibType'
impl From<HiddenType> for MyLibType {
    fn from(it: HiddenType) -> MyLibType {
        match it {}
    }
}

// @!has foo/enum.MyLibType.html '//*[@id="impl-From%3COption%3COption%3COption%3COption%3CHiddenType%3E%3E%3E%3E%3E"]' 'impl From<Option<Option<Option<Option<HiddenType>>>>> for MyLibType'
impl From<Option<Option<Option<Option<HiddenType>>>>> for MyLibType {
    fn from(it: Option<Option<Option<Option<HiddenType>>>>) -> MyLibType {
        todo!()
    }
}
