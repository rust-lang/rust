//! The WIP stable interface to rustc internals.
//!
//! For more information see <https://github.com/rust-lang/project-stable-mir>
//!
//! # Note
//!
//! This API is still completely unstable and subject to change.

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![feature(rustc_private)]
#![feature(ptr_metadata)]
#![feature(type_alias_impl_trait)] // Used to define opaque types.
#![feature(intra_doc_pointers)]

pub mod rustc_internal;

// Make this module private for now since external users should not call these directly.
mod rustc_smir;
