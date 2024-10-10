use rustc_abi::Abi;
use rustc_middle::ty::{self, Instance, InstanceKind, ParamEnv, Ty, TyCtxt};
use rustc_span::def_id::DefId;
use rustc_span::{Span, Symbol};
use rustc_target::abi::call::{FnAbi, PassMode};

use crate::errors::{AbiErrorDisabledVectorTypeCall, AbiErrorDisabledVectorTypeDef};

// Represents the least-constraining feature that is required for vector types up to a certain size
// to have their "proper" ABI.
const X86_VECTOR_FEATURES: &'static [(u64, &'static str)] =
    &[(128, "sse"), (256, "avx"), (512, "avx512f")];

fn do_check_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    abi: &FnAbi<'tcx, Ty<'tcx>>,
    target_feature_def: DefId,
    emit_err: impl Fn(&'static str),
) {
    let feature_def = if tcx.sess.target.arch == "x86" || tcx.sess.target.arch == "x86_64" {
        X86_VECTOR_FEATURES
    } else if tcx.sess.target.arch == "aarch64" {
        // ABI on aarch64 does not depend on target features.
        return;
    } else {
        // FIXME: add support for non-tier1 architectures
        return;
    };
    let codegen_attrs = tcx.codegen_fn_attrs(target_feature_def);
    for arg_abi in abi.args.iter().chain(std::iter::once(&abi.ret)) {
        let size = arg_abi.layout.size;
        if matches!(arg_abi.layout.abi, Abi::Vector { .. })
            && !matches!(arg_abi.mode, PassMode::Indirect { .. })
        {
            let feature = match feature_def.iter().find(|(bits, _)| size.bits() <= *bits) {
                Some((_, feature)) => feature,
                None => panic!("Unknown vector size: {}; arg = {:?}", size.bits(), arg_abi),
            };
            let feature_sym = Symbol::intern(feature);
            if !tcx.sess.unstable_target_features.contains(&feature_sym)
                && !codegen_attrs.target_features.iter().any(|x| x.name == feature_sym)
            {
                emit_err(feature);
            }
        }
    }
}

/// Checks that the ABI of a given instance of a function does not contain vector-passed arguments
/// or return values for which the corresponding target feature is not enabled.
pub fn check_instance_abi<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
    let param_env = ParamEnv::reveal_all();
    let Ok(abi) = tcx.fn_abi_of_instance(param_env.and((instance, ty::List::empty()))) else {
        // An error will be reported during codegen if we cannot determine the ABI of this
        // function.
        return;
    };
    do_check_abi(tcx, abi, instance.def_id(), |required_feature| {
        tcx.dcx().emit_err(AbiErrorDisabledVectorTypeDef {
            span: tcx.def_span(instance.def_id()),
            required_feature,
        });
    })
}

/// Checks that a call expression does not try to pass a vector-passed argument which requires a
/// target feature that the caller does not have, as doing so causes UB because of ABI mismatch.
pub fn check_call_site_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    span: Span,
    caller: InstanceKind<'tcx>,
) {
    let param_env = ParamEnv::reveal_all();
    let callee_abi = match *ty.kind() {
        ty::FnPtr(..) => tcx.fn_abi_of_fn_ptr(param_env.and((ty.fn_sig(tcx), ty::List::empty()))),
        ty::FnDef(def_id, args) => {
            // Intrinsics are handled separately by the compiler.
            if tcx.intrinsic(def_id).is_some() {
                return;
            }
            let instance = ty::Instance::expect_resolve(tcx, param_env, def_id, args, span);
            tcx.fn_abi_of_instance(param_env.and((instance, ty::List::empty())))
        }
        _ => {
            panic!("Invalid function call");
        }
    };

    let Ok(callee_abi) = callee_abi else {
        // ABI failed to compute; this will not get through codegen.
        return;
    };
    do_check_abi(tcx, callee_abi, caller.def_id(), |required_feature| {
        tcx.dcx().emit_err(AbiErrorDisabledVectorTypeCall { span, required_feature });
    })
}
