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
extern crate hidden as nonlocal;

#[doc(hidden)] pub fn local_public_hidden_fn() {}
#[doc(hidden)] fn local_private_hidden_fn() {}
pub fn local_public_non_hidden_fn() {}
fn local_private_non_hidden_fn() {}

// FIXME: lint for items that are within hidden modules also
