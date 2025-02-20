//! This module ensures that if a function's ABI requires a particular target feature,
//! that target feature is enabled both on the callee and all callers.
use rustc_abi::{BackendRepr, ExternAbi, RegKind};
use rustc_hir::CRATE_HIR_ID;
use rustc_middle::mir::{self, traversal};
use rustc_middle::ty::{self, Instance, InstanceKind, Ty, TyCtxt};
use rustc_session::lint::builtin::ABI_UNSUPPORTED_VECTOR_TYPES;
use rustc_span::def_id::DefId;
use rustc_span::{DUMMY_SP, Span, Symbol};
use rustc_target::callconv::{FnAbi, PassMode};

use crate::errors::{
    AbiErrorDisabledVectorTypeCall, AbiErrorDisabledVectorTypeDef,
    AbiErrorUnsupportedVectorTypeCall, AbiErrorUnsupportedVectorTypeDef,
};

fn uses_vector_registers(mode: &PassMode, repr: &BackendRepr) -> bool {
    match mode {
        PassMode::Ignore | PassMode::Indirect { .. } => false,
        PassMode::Cast { pad_i32: _, cast } => {
            cast.prefix.iter().any(|r| r.is_some_and(|x| x.kind == RegKind::Vector))
                || cast.rest.unit.kind == RegKind::Vector
        }
        PassMode::Direct(..) | PassMode::Pair(..) => matches!(repr, BackendRepr::Vector { .. }),
    }
}

/// Checks whether a certain function ABI is compatible with the target features currently enabled
/// for a certain function.
/// If not, `emit_err` is called, with `Some(feature)` if a certain feature should be enabled and
/// with `None` if no feature is known that would make the ABI compatible.
fn do_check_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    abi: &FnAbi<'tcx, Ty<'tcx>>,
    target_feature_def: DefId,
    mut emit_err: impl FnMut(Ty<'tcx>, Option<&'static str>),
) {
    let feature_def = tcx.sess.target.features_for_correct_vector_abi();
    let codegen_attrs = tcx.codegen_fn_attrs(target_feature_def);
    for arg_abi in abi.args.iter().chain(std::iter::once(&abi.ret)) {
        let size = arg_abi.layout.size;
        if uses_vector_registers(&arg_abi.mode, &arg_abi.layout.backend_repr) {
            // Find the first feature that provides at least this vector size.
            let feature = match feature_def.iter().find(|(bits, _)| size.bits() <= *bits) {
                Some((_, feature)) => feature,
                None => {
                    emit_err(arg_abi.layout.ty, None);
                    continue;
                }
            };
            let feature_sym = Symbol::intern(feature);
            if !tcx.sess.unstable_target_features.contains(&feature_sym)
                && !codegen_attrs.target_features.iter().any(|x| x.name == feature_sym)
            {
                emit_err(arg_abi.layout.ty, Some(&feature));
            }
        }
    }
}

/// Checks that the ABI of a given instance of a function does not contain vector-passed arguments
/// or return values for which the corresponding target feature is not enabled.
fn check_instance_abi<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
    let typing_env = ty::TypingEnv::fully_monomorphized();
    let Ok(abi) = tcx.fn_abi_of_instance(typing_env.as_query_input((instance, ty::List::empty())))
    else {
        // An error will be reported during codegen if we cannot determine the ABI of this
        // function.
        return;
    };
    do_check_abi(tcx, abi, instance.def_id(), |ty, required_feature| {
        let span = tcx.def_span(instance.def_id());
        if let Some(required_feature) = required_feature {
            tcx.emit_node_span_lint(
                ABI_UNSUPPORTED_VECTOR_TYPES,
                CRATE_HIR_ID,
                span,
                AbiErrorDisabledVectorTypeDef { span, required_feature, ty },
            );
        } else {
            tcx.emit_node_span_lint(
                ABI_UNSUPPORTED_VECTOR_TYPES,
                CRATE_HIR_ID,
                span,
                AbiErrorUnsupportedVectorTypeDef { span, ty },
            );
        }
    })
}

/// Checks that a call expression does not try to pass a vector-passed argument which requires a
/// target feature that the caller does not have, as doing so causes UB because of ABI mismatch.
fn check_call_site_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    callee: Ty<'tcx>,
    span: Span,
    caller: InstanceKind<'tcx>,
) {
    if callee.fn_sig(tcx).abi() == ExternAbi::Rust {
        // "Rust" ABI never passes arguments in vector registers.
        return;
    }
    let typing_env = ty::TypingEnv::fully_monomorphized();
    let callee_abi = match *callee.kind() {
        ty::FnPtr(..) => {
            tcx.fn_abi_of_fn_ptr(typing_env.as_query_input((callee.fn_sig(tcx), ty::List::empty())))
        }
        ty::FnDef(def_id, args) => {
            // Intrinsics are handled separately by the compiler.
            if tcx.intrinsic(def_id).is_some() {
                return;
            }
            let instance = ty::Instance::expect_resolve(tcx, typing_env, def_id, args, DUMMY_SP);
            tcx.fn_abi_of_instance(typing_env.as_query_input((instance, ty::List::empty())))
        }
        _ => {
            panic!("Invalid function call");
        }
    };

    let Ok(callee_abi) = callee_abi else {
        // ABI failed to compute; this will not get through codegen.
        return;
    };
    do_check_abi(tcx, callee_abi, caller.def_id(), |ty, required_feature| {
        if let Some(required_feature) = required_feature {
            tcx.emit_node_span_lint(
                ABI_UNSUPPORTED_VECTOR_TYPES,
                CRATE_HIR_ID,
                span,
                AbiErrorDisabledVectorTypeCall { span, required_feature, ty },
            );
        } else {
            tcx.emit_node_span_lint(
                ABI_UNSUPPORTED_VECTOR_TYPES,
                CRATE_HIR_ID,
                span,
                AbiErrorUnsupportedVectorTypeCall { span, ty },
            );
        }
    });
}

fn check_callees_abi<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>, body: &mir::Body<'tcx>) {
    // Check all function call terminators.
    for (bb, _data) in traversal::mono_reachable(body, tcx, instance) {
        let terminator = body.basic_blocks[bb].terminator();
        match terminator.kind {
            mir::TerminatorKind::Call { ref func, ref fn_span, .. }
            | mir::TerminatorKind::TailCall { ref func, ref fn_span, .. } => {
                let callee_ty = func.ty(body, tcx);
                let callee_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    ty::TypingEnv::fully_monomorphized(),
                    ty::EarlyBinder::bind(callee_ty),
                );
                check_call_site_abi(tcx, callee_ty, *fn_span, body.source.instance);
            }
            _ => {}
        }
    }
}

pub(crate) fn check_feature_dependent_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &'tcx mir::Body<'tcx>,
) {
    check_instance_abi(tcx, instance);
    check_callees_abi(tcx, instance, body);
}
