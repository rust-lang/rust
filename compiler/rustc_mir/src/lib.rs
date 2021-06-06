/*!

Rust MIR: a lowered representation of Rust.

*/

#![feature(nll)]
#![feature(in_band_lifetimes)]
#![feature(array_windows)]
#![feature(assert_matches)]
#![feature(bindings_after_at)]
#![feature(bool_to_option)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(crate_visibility_modifier)]
#![feature(decl_macro)]
#![feature(exact_size_is_empty)]
#![feature(format_args_capture)]
#![feature(iter_zip)]
#![feature(never_type)]
#![feature(map_try_insert)]
#![feature(min_specialization)]
#![feature(slice_ptr_get)]
#![feature(trusted_len)]
#![feature(try_blocks)]
#![feature(associated_type_defaults)]
#![feature(stmt_expr_attributes)]
#![feature(trait_alias)]
#![feature(option_get_or_insert_default)]
#![feature(once_cell)]
#![feature(control_flow_enum)]
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
    providers.mir_callgraph_reachable = transform::inline::cycle::mir_callgraph_reachable;
    providers.mir_inliner_callees = transform::inline::cycle::mir_inliner_callees;
    providers.destructure_const = |tcx, param_env_and_value| {
        let (param_env, value) = param_env_and_value.into_parts();
        const_eval::destructure_const(tcx, param_env, value)
    };
    providers.const_to_valtree = |tcx, param_env_and_value| {
        let (param_env, raw) = param_env_and_value.into_parts();
        const_eval::const_to_valtree(tcx, param_env, raw)
    };
    providers.deref_const = |tcx, param_env_and_value| {
        let (param_env, value) = param_env_and_value.into_parts();
        const_eval::deref_const(tcx, param_env, value)
    };
}
