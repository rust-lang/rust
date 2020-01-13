//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]
#![feature(bool_to_option)]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![feature(slice_patterns)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc;
#[macro_use]
extern crate log;

use rustc::ty::query::Providers;

mod ty;

pub fn provide(providers: &mut Providers<'_>) {
    ty::provide(providers);
}
