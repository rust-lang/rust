//@ compile-flags: --document-hidden-items
#![no_std]

//@ is "$.index[?(@.name=='func')].attrs" '[{"other": "#[doc(hidden)]"}]'
#[doc(hidden)]
pub fn func() {}

//@ is "$.index[?(@.name=='Unit')].attrs" '[{"other": "#[doc(hidden)]"}]'
#[doc(hidden)]
pub struct Unit;

//@ is "$.index[?(@.name=='hidden')].attrs" '[{"other": "#[doc(hidden)]"}]'
#[doc(hidden)]
pub mod hidden {
    //@ is "$.index[?(@.name=='Inner')].attrs" '[]'
    pub struct Inner;
}
