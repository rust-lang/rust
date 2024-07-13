use rustc_abi::Abi;
use rustc_middle::ty::{self, Instance, InstanceKind, ParamEnv, Ty, TyCtxt};
use rustc_span::{def_id::DefId, Span, Symbol};
use rustc_target::abi::call::{FnAbi, PassMode};

use crate::errors::{AbiErrorDisabledVectorTypeCall, AbiErrorDisabledVectorTypeDef};

const SSE_FEATURES: &'static [&'static str] = &["sse", "sse2", "ssse3", "sse3", "sse4.1", "sse4.2"];
const AVX_FEATURES: &'static [&'static str] = &["avx", "avx2", "f16c", "fma"];
const AVX512_FEATURES: &'static [&'static str] = &[
    "avx512f",
    "avx512bw",
    "avx512cd",
    "avx512er",
    "avx512pf",
    "avx512vl",
    "avx512dq",
    "avx512ifma",
    "avx512vbmi",
    "avx512vnni",
    "avx512bitalg",
    "avx512vpopcntdq",
    "avx512bf16",
    "avx512vbmi2",
];

fn do_check_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    abi: &FnAbi<'tcx, Ty<'tcx>>,
    target_feature_def: DefId,
    emit_err: impl Fn(&'static str),
) {
    // FIXME: add support for other architectures
    if tcx.sess.target.arch != "x86" && tcx.sess.target.arch != "x86_64" {
        return;
    }
    let codegen_attrs = tcx.codegen_fn_attrs(target_feature_def);
    for arg_abi in abi.args.iter().chain(std::iter::once(&abi.ret)) {
        let size = arg_abi.layout.size;
        if matches!(arg_abi.layout.abi, Abi::Vector { .. })
            && matches!(arg_abi.mode, PassMode::Direct(_))
        {
            let features: &[_] = match size.bits() {
                x if x <= 128 => &[SSE_FEATURES, AVX_FEATURES, AVX512_FEATURES],
                x if x <= 256 => &[AVX_FEATURES, AVX512_FEATURES],
                x if x <= 512 => &[AVX512_FEATURES],
                _ => {
                    panic!("Unknown vector size for x86: {}; arg = {:?}", size.bits(), arg_abi)
                }
            };
            let required_feature = features.iter().map(|x| x.iter()).flatten().next().unwrap();
            if !features.iter().map(|x| x.iter()).flatten().any(|feature| {
                let required_feature_sym = Symbol::intern(feature);
                tcx.sess.unstable_target_features.contains(&required_feature_sym)
                    || codegen_attrs.target_features.contains(&required_feature_sym)
            }) {
                emit_err(required_feature);
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
        ty::FnPtr(sig) => tcx.fn_abi_of_fn_ptr(param_env.and((sig, ty::List::empty()))),
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
