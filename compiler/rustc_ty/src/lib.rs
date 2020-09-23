//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(nll)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate tracing;

use rustc_middle::ty::query::Providers;

mod common_traits;
pub mod instance;
mod needs_drop;
mod ty;

pub fn provide(providers: &mut Providers) {
    common_traits::provide(providers);
    needs_drop::provide(providers);
    ty::provide(providers);
    instance::provide(providers);
}
