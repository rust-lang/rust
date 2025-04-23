//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(associated_type_defaults)]
#![feature(box_patterns)]
#![feature(if_let_guard)]
#![feature(iterator_try_collect)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

use rustc_middle::query::Providers;

mod abi;
mod assoc;
mod common_traits;
mod consts;
mod errors;
mod implied_bounds;
mod instance;
mod layout;
mod needs_drop;
mod opaque_types;
mod representability;
pub mod sig_types;
mod stalled_generators;
mod structural_match;
mod ty;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub fn provide(providers: &mut Providers) {
    abi::provide(providers);
    assoc::provide(providers);
    common_traits::provide(providers);
    consts::provide(providers);
    implied_bounds::provide(providers);
    layout::provide(providers);
    needs_drop::provide(providers);
    opaque_types::provide(providers);
    representability::provide(providers);
    ty::provide(providers);
    instance::provide(providers);
    structural_match::provide(providers);
    stalled_generators::provide(providers);
}
