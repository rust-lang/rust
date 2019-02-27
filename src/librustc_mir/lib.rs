/*!

Rust MIR: a lowered representation of Rust. Also: an experiment!

*/

#![feature(nll)]
#![feature(in_band_lifetimes)]
#![feature(slice_patterns)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(crate_visibility_modifier)]
#![feature(core_intrinsics)]
#![feature(const_fn)]
#![feature(decl_macro)]
#![feature(exhaustive_patterns)]
#![feature(range_contains)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_attrs)]
#![feature(never_type)]
#![feature(specialization)]
#![feature(try_trait)]
#![feature(unicode_internals)]
#![feature(step_trait)]
#![feature(slice_concat_ext)]
#![feature(reverse_bits)]
#![feature(try_blocks)]

#![recursion_limit="256"]

#![deny(rust_2018_idioms)]
#![allow(explicit_outlives_requirements)]

#[macro_use] extern crate log;
#[macro_use]
extern crate rustc;
#[macro_use] extern crate rustc_data_structures;
#[allow(unused_extern_crates)]
extern crate serialize as rustc_serialize; // used by deriving
#[macro_use]
extern crate syntax;

mod diagnostics;

mod borrow_check;
mod build;
mod dataflow;
mod hair;
mod lints;
mod shim;
pub mod transform;
pub mod util;
pub mod interpret;
pub mod monomorphize;
pub mod const_eval;

pub use hair::pattern::check_crate as matchck_crate;
use rustc::ty::query::Providers;

pub fn provide(providers: &mut Providers<'_>) {
    borrow_check::provide(providers);
    shim::provide(providers);
    transform::provide(providers);
    monomorphize::partitioning::provide(providers);
    providers.const_eval = const_eval::const_eval_provider;
    providers.const_eval_raw = const_eval::const_eval_raw_provider;
    providers.check_match = hair::pattern::check_match;
}

__build_diagnostic_array! { librustc_mir, DIAGNOSTICS }
