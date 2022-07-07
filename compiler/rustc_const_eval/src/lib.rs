/*!

Rust MIR: a lowered representation of Rust.

*/

#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(control_flow_enum)]
#![feature(decl_macro)]
#![feature(exact_size_is_empty)]
#![feature(let_chains)]
#![feature(let_else)]
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
#![feature(is_some_with)]
#![recursion_limit = "256"]
#![allow(rustc::potential_query_instability)]

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
}

use crate::const_eval::CompileTimeInterpreter;
use crate::interpret::{InterpCx, MemoryKind, OpTy};
use rustc_middle::ty::{layout::TyAndLayout, ParamEnv, TyCtxt};
use rustc_session::Limit;
use rustc_span::Span;

pub fn is_uninit_valid<'tcx>(tcx: TyCtxt<'tcx>, root_span: Span, ty: TyAndLayout<'tcx>) -> bool {
    let machine = CompileTimeInterpreter::new(Limit::new(0), false);
    let mut cx = InterpCx::new(tcx, root_span, ParamEnv::reveal_all(), machine);
    let allocated = cx
        .allocate(ty, MemoryKind::Machine(const_eval::MemoryKind::Heap))
        .expect("failed to allocate for uninit check");
    let ot: OpTy<'_, _> = allocated.into();
    cx.validate_operand(&ot).is_ok()
}

pub fn is_zero_valid<'tcx>(tcx: TyCtxt<'tcx>, root_span: Span, ty: TyAndLayout<'tcx>) -> bool {
    let machine = CompileTimeInterpreter::new(Limit::new(0), false);

    let mut cx = InterpCx::new(tcx, root_span, ParamEnv::reveal_all(), machine);

    // We could panic here... Or we could just return "yeah it's valid whatever". Or let
    // codegen_panic_intrinsic return an error that halts compilation.
    // I'm not exactly sure *when* this can fail. OOM?
    let allocated = cx
        .allocate(ty, MemoryKind::Machine(const_eval::MemoryKind::Heap))
        .expect("failed to allocate for uninit check");

    // Again, unclear what to do here if it fails.
    cx.write_bytes_ptr(allocated.ptr, std::iter::repeat(0_u8).take(ty.layout.size().bytes_usize()))
        .expect("failed to write bytes for zero valid check");

    let ot: OpTy<'_, _> = allocated.into();

    // Assume that if it failed, it's a validation failure.
    cx.validate_operand(&ot).is_ok()
}
