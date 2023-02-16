/*!

Rust MIR: a lowered representation of Rust.

*/

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
#![feature(is_some_and)]
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

use rustc_middle::ty;
use rustc_middle::ty::query::Providers;
use rustc_target::abi::InitKind;

pub fn provide(providers: &mut Providers) {
    const_eval::provide(providers);
    providers.eval_to_const_value_raw = const_eval::eval_to_const_value_raw_provider;
    providers.eval_to_allocation_raw = const_eval::eval_to_allocation_raw_provider;
    providers.const_caller_location = const_eval::const_caller_location;
    providers.eval_to_valtree = |tcx, param_env_and_value| {
        let (param_env, raw) = param_env_and_value.into_parts();
        const_eval::eval_to_valtree(tcx, param_env, raw)
    };
    providers.try_destructure_mir_constant = |tcx, param_env_and_value| {
        let (param_env, value) = param_env_and_value.into_parts();
        const_eval::try_destructure_mir_constant(tcx, param_env, value).ok()
    };
    providers.valtree_to_const_val = |tcx, (ty, valtree)| {
        const_eval::valtree_to_const_value(tcx, ty::ParamEnv::empty().and(ty), valtree)
    };
    providers.deref_mir_constant = |tcx, param_env_and_value| {
        let (param_env, value) = param_env_and_value.into_parts();
        const_eval::deref_mir_constant(tcx, param_env, value)
    };
    providers.permits_uninit_init = |tcx, param_env_and_ty| {
        util::might_permit_raw_init(tcx, param_env_and_ty, InitKind::UninitMitigated0x01Fill)
    };
    providers.permits_zero_init =
        |tcx, param_env_and_ty| util::might_permit_raw_init(tcx, param_env_and_ty, InitKind::Zero);
}
