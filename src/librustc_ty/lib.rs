//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]
#![feature(bool_to_option)]
#![feature(nll)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc;
#[macro_use]
extern crate log;

use rustc::ty::query::Providers;

mod common_traits;
pub mod instance;
mod needs_drop;
mod ty;

pub fn provide(providers: &mut Providers<'_>) {
    common_traits::provide(providers);
    needs_drop::provide(providers);
    ty::provide(providers);
}
