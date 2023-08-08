/*!

Rust MIR: a lowered representation of Rust.

*/

#![deny(rustc::untranslatable_diagnostic)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(decl_macro)]
#![feature(exact_size_is_empty)]
#![feature(let_chains)]
#![feature(map_try_insert)]
#![feature(min_specialization)]
#![feature(slice_ptr_get)]
#![feature(option_get_or_insert_default)]
#![feature(never_type)]
#![feature(trait_alias)]
#![feature(trusted_len)]
#![feature(trusted_step)]
#![feature(try_blocks)]
#![feature(yeet_expr)]
#![feature(if_let_guard)]
#![recursion_limit = "256"]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

pub mod const_eval;
mod errors;
pub mod interpret;
pub mod transform;
pub mod util;

pub use errors::ReportErrorExt;

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;
use rustc_middle::query::Providers;
use rustc_middle::ty;

fluent_messages! { "../messages.ftl" }

pub fn provide(providers: &mut Providers) {
    const_eval::provide(providers);
    providers.eval_to_const_value_raw = const_eval::eval_to_const_value_raw_provider;
    providers.eval_to_allocation_raw = const_eval::eval_to_allocation_raw_provider;
    providers.const_caller_location = const_eval::const_caller_location;
    providers.eval_to_valtree = |tcx, param_env_and_value| {
        let (param_env, raw) = param_env_and_value.into_parts();
        const_eval::eval_to_valtree(tcx, param_env, raw)
    };
    providers.try_destructure_mir_constant_for_diagnostics =
        |tcx, (cv, ty)| const_eval::try_destructure_mir_constant_for_diagnostics(tcx, cv, ty);
    providers.valtree_to_const_val = |tcx, (ty, valtree)| {
        const_eval::valtree_to_const_value(tcx, ty::ParamEnv::empty().and(ty), valtree)
    };
    providers.check_validity_requirement = |tcx, (init_kind, param_env_and_ty)| {
        util::check_validity_requirement(tcx, init_kind, param_env_and_ty)
    };
}
