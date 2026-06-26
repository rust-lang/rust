//@ compile-flags: --document-hidden-items

//@ has "$.index[?(@.name == 'HiddenPubStruct')]"
//@ has "$.index[?(@.inner.impl)]"
//@ has "$.index[?(@.name=='PubTrait')]"
pub trait PubTrait {}

#[doc(hidden)]
pub struct HiddenPubStruct;

impl PubTrait for HiddenPubStruct {}
