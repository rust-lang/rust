/*!

Rust MIR: a lowered representation of Rust. Also: an experiment!

*/

#![feature(nll)]
#![feature(in_band_lifetimes)]
#![feature(inner_deref)]
#![feature(slice_patterns)]
#![feature(bool_to_option)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(crate_visibility_modifier)]
#![feature(core_intrinsics)]
#![feature(const_fn)]
#![feature(decl_macro)]
#![feature(drain_filter)]
#![feature(exhaustive_patterns)]
#![feature(iter_order_by)]
#![feature(never_type)]
#![feature(specialization)]
#![feature(try_trait)]
#![feature(unicode_internals)]
#![feature(slice_concat_ext)]
#![feature(trusted_len)]
#![feature(try_blocks)]
#![feature(associated_type_bounds)]
#![feature(range_is_empty)]
#![feature(stmt_expr_attributes)]
#![feature(trait_alias)]
#![feature(matches_macro)]

#![recursion_limit="256"]

#[macro_use] extern crate log;
#[macro_use] extern crate rustc;
#[macro_use] extern crate syntax;

mod borrow_check;
mod build;
pub mod dataflow;
mod hair;
mod lints;
mod shim;
pub mod transform;
pub mod util;
pub mod interpret;
pub mod monomorphize;
pub mod const_eval;

use rustc::ty::query::Providers;

pub fn provide(providers: &mut Providers<'_>) {
    borrow_check::provide(providers);
    shim::provide(providers);
    transform::provide(providers);
    monomorphize::partitioning::provide(providers);
    providers.const_eval = const_eval::const_eval_provider;
    providers.const_eval_raw = const_eval::const_eval_raw_provider;
    providers.check_match = hair::pattern::check_match;
    providers.const_caller_location = const_eval::const_caller_location;
    providers.const_field = |tcx, param_env_and_value| {
        let (param_env, (value, field)) = param_env_and_value.into_parts();
        const_eval::const_field(tcx, param_env, None, field, value)
    };
}
