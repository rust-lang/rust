/*!

Rust MIR: a lowered representation of Rust.

*/

#![feature(nll)]
#![feature(in_band_lifetimes)]
#![feature(array_windows)]
#![feature(bindings_after_at)]
#![feature(bool_to_option)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(const_fn)]
#![feature(const_panic)]
#![feature(crate_visibility_modifier)]
#![feature(decl_macro)]
#![feature(exact_size_is_empty)]
#![feature(exhaustive_patterns)]
#![feature(never_type)]
#![feature(min_specialization)]
#![feature(trusted_len)]
#![feature(try_blocks)]
#![feature(associated_type_defaults)]
#![feature(stmt_expr_attributes)]
#![feature(trait_alias)]
#![feature(option_expect_none)]
#![feature(or_patterns)]
#![feature(once_cell)]
#![feature(control_flow_enum)]
#![feature(str_split_once)]
#![recursion_limit = "256"]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

mod borrow_check;
pub mod const_eval;
pub mod dataflow;
pub mod interpret;
pub mod monomorphize;
mod shim;
pub mod transform;
pub mod util;

use rustc_middle::ty::query::Providers;

pub fn provide(providers: &mut Providers) {
    borrow_check::provide(providers);
    const_eval::provide(providers);
    shim::provide(providers);
    transform::provide(providers);
    monomorphize::partitioning::provide(providers);
    monomorphize::polymorphize::provide(providers);
    providers.eval_to_const_value_raw = const_eval::eval_to_const_value_raw_provider;
    providers.eval_to_allocation_raw = const_eval::eval_to_allocation_raw_provider;
    providers.const_caller_location = const_eval::const_caller_location;
    providers.destructure_const = |tcx, param_env_and_value| {
        let (param_env, value) = param_env_and_value.into_parts();
        const_eval::destructure_const(tcx, param_env, value)
    };
    providers.deref_const = |tcx, param_env_and_value| {
        let (param_env, value) = param_env_and_value.into_parts();
        const_eval::deref_const(tcx, param_env, value)
    };
}
