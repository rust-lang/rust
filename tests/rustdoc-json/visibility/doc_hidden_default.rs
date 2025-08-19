#![no_std]

// Without `--document-hidden-items`,
// none of these items are present in rustdoc JSON.

//@ !has "$.index[?(@.name=='func')]"
#[doc(hidden)]
pub fn func() {}

//@ !has "$.index[?(@.name=='Unit')]"
#[doc(hidden)]
pub struct Unit;

//@ !has "$.index[?(@.name=='hidden')]"
#[doc(hidden)]
pub mod hidden {
    //@ !has "$.index[?(@.name=='Inner')]"
    pub struct Inner;
}
