#[cfg(feature="master")]
use gccjit::FnAttribute;
use gccjit::Function;
use rustc_attr::InstructionSetAttr;
#[cfg(feature="master")]
use rustc_attr::InlineAttr;
use rustc_codegen_ssa::target_features::tied_target_features;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty;
#[cfg(feature="master")]
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_session::Session;
use rustc_span::symbol::sym;
use smallvec::{smallvec, SmallVec};

use crate::{context::CodegenCx, errors::TiedTargetFeatures};

// Given a map from target_features to whether they are enabled or disabled,
// ensure only valid combinations are allowed.
pub fn check_tied_features(sess: &Session, features: &FxHashMap<&str, bool>) -> Option<&'static [&'static str]> {
    for tied in tied_target_features(sess) {
        // Tied features must be set to the same value, or not set at all
        let mut tied_iter = tied.iter();
        let enabled = features.get(tied_iter.next().unwrap());
        if tied_iter.any(|feature| enabled != features.get(feature)) {
            return Some(tied);
        }
    }
    None
}

// TODO(antoyo): maybe move to a new module gcc_util.
// To find a list of GCC's names, check https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
fn to_gcc_features<'a>(sess: &Session, s: &'a str) -> SmallVec<[&'a str; 2]> {
    let arch = if sess.target.arch == "x86_64" { "x86" } else { &*sess.target.arch };
    match (arch, s) {
        ("x86", "sse4.2") => smallvec!["sse4.2", "crc32"],
        ("x86", "pclmulqdq") => smallvec!["pclmul"],
        ("x86", "rdrand") => smallvec!["rdrnd"],
        ("x86", "bmi1") => smallvec!["bmi"],
        ("x86", "cmpxchg16b") => smallvec!["cx16"],
        ("x86", "avx512vaes") => smallvec!["vaes"],
        ("x86", "avx512gfni") => smallvec!["gfni"],
        ("x86", "avx512vpclmulqdq") => smallvec!["vpclmulqdq"],
        // NOTE: seems like GCC requires 'avx512bw' for 'avx512vbmi2'.
        ("x86", "avx512vbmi2") => smallvec!["avx512vbmi2", "avx512bw"],
        // NOTE: seems like GCC requires 'avx512bw' for 'avx512bitalg'.
        ("x86", "avx512bitalg") => smallvec!["avx512bitalg", "avx512bw"],
        ("aarch64", "rcpc2") => smallvec!["rcpc-immo"],
        ("aarch64", "dpb") => smallvec!["ccpp"],
        ("aarch64", "dpb2") => smallvec!["ccdp"],
        ("aarch64", "frintts") => smallvec!["fptoint"],
        ("aarch64", "fcma") => smallvec!["complxnum"],
        ("aarch64", "pmuv3") => smallvec!["perfmon"],
        ("aarch64", "paca") => smallvec!["pauth"],
        ("aarch64", "pacg") => smallvec!["pauth"],
        // Rust ties fp and neon together. In LLVM neon implicitly enables fp,
        // but we manually enable neon when a feature only implicitly enables fp
        ("aarch64", "f32mm") => smallvec!["f32mm", "neon"],
        ("aarch64", "f64mm") => smallvec!["f64mm", "neon"],
        ("aarch64", "fhm") => smallvec!["fp16fml", "neon"],
        ("aarch64", "fp16") => smallvec!["fullfp16", "neon"],
        ("aarch64", "jsconv") => smallvec!["jsconv", "neon"],
        ("aarch64", "sve") => smallvec!["sve", "neon"],
        ("aarch64", "sve2") => smallvec!["sve2", "neon"],
        ("aarch64", "sve2-aes") => smallvec!["sve2-aes", "neon"],
        ("aarch64", "sve2-sm4") => smallvec!["sve2-sm4", "neon"],
        ("aarch64", "sve2-sha3") => smallvec!["sve2-sha3", "neon"],
        ("aarch64", "sve2-bitperm") => smallvec!["sve2-bitperm", "neon"],
        (_, s) => smallvec![s],
    }
}

/// Get GCC attribute for the provided inline heuristic.
#[cfg(feature="master")]
#[inline]
fn inline_attr<'gcc, 'tcx>(cx: &CodegenCx<'gcc, 'tcx>, inline: InlineAttr) -> Option<FnAttribute<'gcc>> {
    match inline {
        InlineAttr::Hint => Some(FnAttribute::Inline),
        InlineAttr::Always => Some(FnAttribute::AlwaysInline),
        InlineAttr::Never => {
            if cx.sess().target.arch != "amdgpu" {
                Some(FnAttribute::NoInline)
            } else {
                None
            }
        }
        InlineAttr::None => None,
    }
}

/// Composite function which sets GCC attributes for function depending on its AST (`#[attribute]`)
/// attributes.
pub fn from_fn_attrs<'gcc, 'tcx>(
    cx: &CodegenCx<'gcc, 'tcx>,
    #[cfg_attr(not(feature="master"), allow(unused_variables))]
    func: Function<'gcc>,
    instance: ty::Instance<'tcx>,
) {
    let codegen_fn_attrs = cx.tcx.codegen_fn_attrs(instance.def_id());

    #[cfg(feature="master")]
    {
        let inline =
            if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
                InlineAttr::Never
            }
            else if codegen_fn_attrs.inline == InlineAttr::None && instance.def.requires_inline(cx.tcx) {
                InlineAttr::Hint
            }
            else {
                codegen_fn_attrs.inline
            };
        if let Some(attr) = inline_attr(cx, inline) {
            func.add_attribute(attr);
        }
    }

    let function_features =
        codegen_fn_attrs.target_features.iter().map(|features| features.as_str()).collect::<Vec<&str>>();

    if let Some(features) = check_tied_features(cx.tcx.sess, &function_features.iter().map(|features| (*features, true)).collect()) {
        let span = cx.tcx
            .get_attr(instance.def_id(), sym::target_feature)
            .map_or_else(|| cx.tcx.def_span(instance.def_id()), |a| a.span);
        cx.tcx.sess.create_err(TiedTargetFeatures {
            features: features.join(", "),
            span,
        })
            .emit();
        return;
    }

    let mut function_features = function_features
        .iter()
        .flat_map(|feat| to_gcc_features(cx.tcx.sess, feat).into_iter())
        .chain(codegen_fn_attrs.instruction_set.iter().map(|x| match x {
            InstructionSetAttr::ArmA32 => "-thumb-mode", // TODO(antoyo): support removing feature.
            InstructionSetAttr::ArmT32 => "thumb-mode",
        }))
        .collect::<Vec<_>>();

    // TODO(antoyo): check if we really need global backend features. (Maybe they could be applied
    // globally?)
    let mut global_features = cx.tcx.global_backend_features(()).iter().map(|s| s.as_str());
    function_features.extend(&mut global_features);
    let target_features = function_features.join(",");
    if !target_features.is_empty() {
        #[cfg(feature="master")]
        func.add_attribute(FnAttribute::Target(&target_features));
    }
}
