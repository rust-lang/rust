// ignore-tidy-linelength
//@ revisions: noflags doc-priv doc-hidden doc-both
//@ aux-build: hidden.rs
//@ build-aux-docs
//@[doc-priv]compile-flags: --document-private-items
//@[doc-hidden]compile-flags: -Z unstable-options --document-hidden-items
//@[doc-both]compile-flags: -Z unstable-options --document-hidden-items --document-private-items

#![deny(rustdoc::private_intra_doc_links)]
#![deny(rustdoc::hidden_intra_doc_links)]
#![deny(rustdoc::broken_intra_doc_links)]

//! All these links are broken and should produce warnings:
//! * [crate::local_public_hidden_fn]
//~^ ERROR: non-hidden documentation for `link_to_hidden_144664` links to hidden item `crate::local_public_hidden_fn`

// for local items that are hidden and private, both lints are emmitted
//! * [crate::local_private_hidden_fn]
//~^ ERROR: public documentation for `link_to_hidden_144664` links to private item `crate::local_private_hidden_fn`
//~| ERROR: non-hidden documentation for `link_to_hidden_144664` links to hidden item `crate::local_private_hidden_fn`

//! * [crate::local_private_non_hidden_fn]
//~^ ERROR: public documentation for `link_to_hidden_144664` links to private item `crate::local_private_non_hidden_fn`
//! * [nonlocal::public_hidden_fn]
//~^ ERROR: non-hidden documentation for `link_to_hidden_144664` links to hidden item `nonlocal::public_hidden_fn`

// for cross-crate private items, rustdoc doesn't even know if they exist, so we get a generic error.
//! * [nonlocal::private_hidden_fn]
//~^ ERROR: unresolved link
//! * [nonlocal::private_non_hidden_fn]
//~^ ERROR: unresolved link
//! * [crate::hidden_mod::a]
//~^ ERROR: unresolved link
//! * [crate::hidden_mod::b]
//~^ ERROR: non-hidden documentation for `link_to_hidden_144664` links to hidden item `crate::hidden_mod::b`
//! * [crate::hidden_mod::c]
//~^ ERROR: non-hidden documentation for `link_to_hidden_144664` links to hidden item `crate::hidden_mod::c`
extern crate hidden as nonlocal;

#[doc(hidden)] pub fn local_public_hidden_fn() {}
/// [Private hidden to public hidden is fine](crate::local_public_hidden_fn)
#[doc(hidden)] fn local_private_hidden_fn() {}
pub fn local_public_non_hidden_fn() {}
fn local_private_non_hidden_fn() {}

pub struct PublicStruct;

#[doc(hidden)]
pub mod hidden_mod {
    /// [This link is fine, both are indirectly hidden](self::b)
    fn a() {}
    /// [This link also fine, both are hidden, doesn't matter that only one is indirect](self::c)
    pub fn b() {}
    /// [Hidden docs can link to non-hidden](crate::local_public_non_hidden_fn)
    #[doc(hidden)]
    pub fn c() {}

    #[doc(hidden)]
    pub struct HiddenStruct;


    impl crate::PublicNonHiddenTrait for HiddenStruct {
        /// This method is actually hidden,
        /// as the struct is hidden and thus it will not show up in docs.
        /// [This link is ok](self::c)
        fn non_hidden_trait_method(&self) {}
    }

    pub trait PublicHiddenTrait {
        fn hidden_trait_method(&self);
    }
}

pub trait PublicNonHiddenTrait {
    fn non_hidden_trait_method(&self);
}

impl crate::hidden_mod::PublicHiddenTrait for PublicStruct {
    /// This method is hidden as the trait is hidden.
    /// [This link is ok](crate::local_public_hidden_fn)
    fn hidden_trait_method(&self) {}
}
