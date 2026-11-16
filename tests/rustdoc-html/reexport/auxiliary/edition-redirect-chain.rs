//@ edition: 2024
//@ aux-crate: edition_redirect_target=edition-redirect-target.rs

#![feature(rustc_attrs)]
#![allow(internal_features)]

/// Default import documentation.
pub use edition_redirect_target::Current as Item;

/// Redirect documentation.
#[rustc_edition_redirect = "..=2021"]
pub use edition_redirect_target::Intermediate as Item;
