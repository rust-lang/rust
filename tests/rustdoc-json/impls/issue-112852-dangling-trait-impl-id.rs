#![feature(no_core)]
#![no_core]

// @count "$.index[*][?(@.inner.impl)]" 1
// @!has "$.index[*][?(@.name == 'HiddenPubStruct')]"
// @has "$.index[*][?(@.name == 'NotHiddenPubStruct')]"
// @has "$.index[*][?(@.name=='PubTrait')]"
pub trait PubTrait {}

#[doc(hidden)]
pub mod hidden {
    pub struct HiddenPubStruct;

    impl crate::PubTrait for HiddenPubStruct {}
}

pub mod not_hidden {
    pub struct NotHiddenPubStruct;

    impl crate::PubTrait for NotHiddenPubStruct {}
}
