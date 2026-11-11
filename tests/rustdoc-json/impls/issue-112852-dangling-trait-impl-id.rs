//@ has "$.index[?(@.name=='PubTrait')]"
pub trait PubTrait {}

#[doc(hidden)]
pub mod hidden {
    //@ !has "$.index[?(@.name == 'HiddenPubStruct')]"
    pub struct HiddenPubStruct;

    //@ !has "$.index[?(@.docs == 'Not Here')]"
    /// Not Here
    impl crate::PubTrait for HiddenPubStruct {}
}

pub mod not_hidden {
    //@ has "$.index[?(@.name == 'NotHiddenPubStruct')]"
    pub struct NotHiddenPubStruct;

    //@ has "$.index[?(@.docs == 'Here')]"
    /// Here
    impl crate::PubTrait for NotHiddenPubStruct {}
}
