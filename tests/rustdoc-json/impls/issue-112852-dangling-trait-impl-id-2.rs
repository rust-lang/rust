#![feature(no_core)]
#![no_core]

// @count "$.index[*][?(@.inner.impl)]" 1
// @!has "$.index[*][?(@.name == 'HiddenPubStruct')]"
// @has "$.index[*][?(@.name == 'NotHiddenPubStruct')]"
// @has "$.index[*][?(@.name=='PubTrait')]"
pub trait PubTrait {}

#[doc(hidden)]
pub struct HiddenPubStruct;
pub struct NotHiddenPubStruct;

impl PubTrait for HiddenPubStruct {}
impl PubTrait for NotHiddenPubStruct {}
