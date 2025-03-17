//! This module ensures that if a function's ABI requires a particular target feature,
//! that target feature is enabled both on the callee and all callers.
use rustc_abi::{BackendRepr, RegKind};
use rustc_hir::CRATE_HIR_ID;
use rustc_middle::mir::{self, traversal};
use rustc_middle::ty::layout::LayoutCx;
use rustc_middle::ty::{self, Instance, InstanceKind, Ty, TyCtxt, TypingEnv};
use rustc_session::lint::builtin::{ABI_UNSUPPORTED_VECTOR_TYPES, WASM_C_ABI};
use rustc_span::def_id::DefId;
use rustc_span::{DUMMY_SP, Span, Symbol, sym};
use rustc_target::callconv::{ArgAbi, Conv, FnAbi, PassMode};
use rustc_target::spec::{HasWasmCAbiOpt, WasmCAbi};

use crate::errors;

fn uses_vector_registers(mode: &PassMode, repr: &BackendRepr) -> bool {
    match mode {
        PassMode::Ignore | PassMode::Indirect { .. } => false,
        PassMode::Cast { pad_i32: _, cast } => {
            cast.prefix.iter().any(|r| r.is_some_and(|x| x.kind == RegKind::Vector))
                || cast.rest.unit.kind == RegKind::Vector
        }
        PassMode::Direct(..) | PassMode::Pair(..) => matches!(repr, BackendRepr::SimdVector { .. }),
    }
}

/// Checks whether a certain function ABI is compatible with the target features currently enabled
/// for a certain function.
/// `is_call` indicates whether this is a call-site check or a definition-site check;
/// this is only relevant for the wording in the emitted error.
fn do_check_simd_vector_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    abi: &FnAbi<'tcx, Ty<'tcx>>,
    def_id: DefId,
    is_call: bool,
    span: impl Fn() -> Span,
) {
    // We check this on all functions, including those using the "Rust" ABI.
    // For the "Rust" ABI it would be a bug if the lint ever triggered, but better safe than sorry.
    let feature_def = tcx.sess.target.features_for_correct_vector_abi();
    let codegen_attrs = tcx.codegen_fn_attrs(def_id);
    let have_feature = |feat: Symbol| {
        tcx.sess.unstable_target_features.contains(&feat)
            || codegen_attrs.target_features.iter().any(|x| x.name == feat)
    };
    for arg_abi in abi.args.iter().chain(std::iter::once(&abi.ret)) {
        let size = arg_abi.layout.size;
        if uses_vector_registers(&arg_abi.mode, &arg_abi.layout.backend_repr) {
            // Find the first feature that provides at least this vector size.
            let feature = match feature_def.iter().find(|(bits, _)| size.bits() <= *bits) {
                Some((_, feature)) => feature,
                None => {
                    let span = span();
                    tcx.emit_node_span_lint(
                        ABI_UNSUPPORTED_VECTOR_TYPES,
                        CRATE_HIR_ID,
                        span,
                        errors::AbiErrorUnsupportedVectorType {
                            span,
                            ty: arg_abi.layout.ty,
                            is_call,
                        },
                    );
                    continue;
                }
            };
            if !have_feature(Symbol::intern(feature)) {
                // Emit error.
                let span = span();
                tcx.emit_node_span_lint(
                    ABI_UNSUPPORTED_VECTOR_TYPES,
                    CRATE_HIR_ID,
                    span,
                    errors::AbiErrorDisabledVectorType {
                        span,
                        required_feature: feature,
                        ty: arg_abi.layout.ty,
                        is_call,
                    },
                );
            }
        }
    }
    // The `vectorcall` ABI is special in that it requires SSE2 no matter which types are being passed.
    if abi.conv == Conv::X86VectorCall && !have_feature(sym::sse2) {
        tcx.dcx().emit_err(errors::AbiRequiredTargetFeature {
            span: span(),
            required_feature: "sse2",
            abi: "vectorcall",
            is_call,
        });
    }
}

fn wasm_abi_safe<'tcx>(tcx: TyCtxt<'tcx>, arg: &ArgAbi<'tcx, Ty<'tcx>>) -> bool {
    if matches!(arg.layout.backend_repr, BackendRepr::Scalar(_)) {
        return true;
    }

    // This matches `unwrap_trivial_aggregate` in the wasm ABI logic.
    if arg.layout.is_aggregate() {
        let cx = LayoutCx::new(tcx, TypingEnv::fully_monomorphized());
        if let Some(unit) = arg.layout.homogeneous_aggregate(&cx).ok().and_then(|ha| ha.unit()) {
            let size = arg.layout.size;
            // Ensure there's just a single `unit` element in `arg`.
            if unit.size == size {
                return true;
            }
        }
    }

    false
}

/// Warns against usage of `extern "C"` on wasm32-unknown-unknown that is affected by the
/// ABI transition.
fn do_check_wasm_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    abi: &FnAbi<'tcx, Ty<'tcx>>,
    is_call: bool,
    span: impl Fn() -> Span,
) {
    // Only proceed for `extern "C" fn` on wasm32-unknown-unknown (same check as what `adjust_for_foreign_abi` uses to call `compute_wasm_abi_info`).
    if !(tcx.sess.target.arch == "wasm32"
        && tcx.sess.target.os == "unknown"
        && tcx.wasm_c_abi_opt() == WasmCAbi::Legacy
        && abi.conv == Conv::C)
    {
        return;
    }
    // Warn against all types whose ABI will change. That's all arguments except for things passed as scalars.
    // Return values are not affected by this change.
    for arg_abi in abi.args.iter() {
        if wasm_abi_safe(tcx, arg_abi) {
            continue;
        }
        let span = span();
        tcx.emit_node_span_lint(
            WASM_C_ABI,
            CRATE_HIR_ID,
            span,
            errors::WasmCAbiTransition { ty: arg_abi.layout.ty, is_call },
        );
        // Let's only warn once per function.
        break;
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
    do_check_simd_vector_abi(tcx, abi, instance.def_id(), /*is_call*/ false, || {
        tcx.def_span(instance.def_id())
    });
    do_check_wasm_abi(tcx, abi, /*is_call*/ false, || tcx.def_span(instance.def_id()));
}

/// Checks that a call expression does not try to pass a vector-passed argument which requires a
/// target feature that the caller does not have, as doing so causes UB because of ABI mismatch.
fn check_call_site_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    callee: Ty<'tcx>,
    span: Span,
    caller: InstanceKind<'tcx>,
) {
    if callee.fn_sig(tcx).abi().is_rustic_abi() {
        // we directly handle the soundness of Rust ABIs
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
    do_check_simd_vector_abi(tcx, callee_abi, caller.def_id(), /*is_call*/ true, || span);
    do_check_wasm_abi(tcx, callee_abi, /*is_call*/ true, || span);
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
