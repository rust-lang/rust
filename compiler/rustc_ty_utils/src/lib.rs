//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(iterator_try_collect)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(box_patterns)]
#![recursion_limit = "256"]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate tracing;

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;
use rustc_middle::ty::query::Providers;

mod abi;
mod assoc;
mod common_traits;
mod consts;
mod errors;
mod implied_bounds;
pub mod instance;
mod layout;
mod layout_sanity_check;
mod needs_drop;
pub mod representability;
mod structural_match;
mod ty;

fluent_messages! { "../messages.ftl" }

pub fn provide(providers: &mut Providers) {
    abi::provide(providers);
    assoc::provide(providers);
    common_traits::provide(providers);
    consts::provide(providers);
    implied_bounds::provide(providers);
    layout::provide(providers);
    needs_drop::provide(providers);
    representability::provide(providers);
    ty::provide(providers);
    instance::provide(providers);
    structural_match::provide(providers);
}
