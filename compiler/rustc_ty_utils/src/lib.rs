//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![cfg_attr(bootstrap, feature(assert_matches))]
#![cfg_attr(bootstrap, feature(if_let_guard))]
#![feature(associated_type_defaults)]
#![feature(box_patterns)]
#![feature(iterator_try_collect)]
#![feature(never_type)]
// tidy-alphabetical-end

use rustc_middle::util::Providers;

mod abi;
mod assoc;
mod common_traits;
mod consts;
mod errors;
mod implied_bounds;
mod instance;
mod layout;
mod needs_drop;
mod nested_bodies;
mod opaque_types;
mod representability;
pub mod sig_types;
mod structural_match;
mod ty;

pub fn provide(providers: &mut Providers) {
    abi::provide(&mut providers.queries);
    assoc::provide(&mut providers.queries);
    common_traits::provide(&mut providers.queries);
    consts::provide(&mut providers.queries);
    implied_bounds::provide(&mut providers.queries);
    layout::provide(&mut providers.queries);
    needs_drop::provide(&mut providers.queries);
    opaque_types::provide(&mut providers.queries);
    representability::provide(providers);
    ty::provide(&mut providers.queries);
    instance::provide(&mut providers.queries);
    structural_match::provide(&mut providers.queries);
    nested_bodies::provide(&mut providers.queries);
}
