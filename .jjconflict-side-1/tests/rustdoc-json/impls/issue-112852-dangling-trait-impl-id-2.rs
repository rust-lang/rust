//@ has "$.index[?(@.docs=='Here')]"
//@ !has "$.index[?(@.docs=='Not Here')]"
//@ !has "$.index[?(@.name == 'HiddenPubStruct')]"
//@ has "$.index[?(@.name == 'NotHiddenPubStruct')]"
//@ has "$.index[?(@.name=='PubTrait')]"
pub trait PubTrait {}

#[doc(hidden)]
pub struct HiddenPubStruct;
pub struct NotHiddenPubStruct;

/// Not Here
impl PubTrait for HiddenPubStruct {}
/// Here
impl PubTrait for NotHiddenPubStruct {}
