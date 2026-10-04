//@ edition: 2024

pub struct ActuallyPub {}
#[doc(hidden)]
pub struct DocHidden {}

pub mod pub_mod {
    pub struct ActuallyPubInPubMod {}
    #[doc(hidden)]
    pub struct DocHiddenInPubMod {}
}

#[doc(hidden)]
pub mod hidden_mod {
    pub struct ActuallyPubInHiddenMod {}
    #[doc(hidden)]
    pub struct DocHiddenInHiddenMod {}
}
