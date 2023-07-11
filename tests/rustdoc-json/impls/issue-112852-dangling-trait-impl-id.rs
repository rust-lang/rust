#![feature(no_core)]
#![no_core]

// @!has "$.index[*][?(@.name == 'HiddenPubStruct')]"
// @!has "$.index[*][?(@.inner.impl)]"
// @has "$.index[*][?(@.name=='PubTrait')]"
pub trait PubTrait {}

#[doc(hidden)]
pub mod hidden {
    pub struct HiddenPubStruct;

    impl crate::PubTrait for HiddenPubStruct {}
}
