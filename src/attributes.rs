#[cfg(feature="master")]
use gccjit::FnAttribute;
use gccjit::Function;
use rustc_attr::InstructionSetAttr;
#[cfg(feature="master")]
use rustc_attr::InlineAttr;
use rustc_middle::ty;
#[cfg(feature="master")]
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_span::symbol::sym;

use crate::{context::CodegenCx, errors::TiedTargetFeatures};
use crate::gcc_util::{check_tied_features, to_gcc_features};

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
            if let FnAttribute::AlwaysInline = attr {
                func.add_attribute(FnAttribute::Inline);
            }
            func.add_attribute(attr);
        }

        if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::COLD) {
            func.add_attribute(FnAttribute::Cold);
        }
        if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::FFI_PURE) {
            func.add_attribute(FnAttribute::Pure);
        }
        if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::FFI_CONST) {
            func.add_attribute(FnAttribute::Const);
        }
    }

    let function_features =
        codegen_fn_attrs.target_features.iter().map(|features| features.as_str()).collect::<Vec<&str>>();

    if let Some(features) = check_tied_features(cx.tcx.sess, &function_features.iter().map(|features| (*features, true)).collect()) {
        let span = cx.tcx
            .get_attr(instance.def_id(), sym::target_feature)
            .map_or_else(|| cx.tcx.def_span(instance.def_id()), |a| a.span);
        cx.tcx.dcx().create_err(TiedTargetFeatures {
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

    // TODO(antoyo): cg_llvm adds global features to each function so that LTO keep them.
    // Check if GCC requires the same.
    let mut global_features = cx.tcx.global_backend_features(()).iter().map(|s| s.as_str());
    function_features.extend(&mut global_features);
    let target_features = function_features
        .iter()
        .filter_map(|feature| {
            // FIXME(antoyo): for some reasons, disabling SSE results in the following error when
            // compiling Rust for Linux:
            // SSE register return with SSE disabled
            // TODO(antoyo): support soft-float and retpoline-external-thunk.
            if feature.contains("soft-float") || feature.contains("retpoline-external-thunk") || *feature == "-sse" {
                return None;
            }

            if feature.starts_with('-') {
                Some(format!("no{}", feature))
            }
            else if feature.starts_with('+') {
                Some(feature[1..].to_string())
            }
            else {
                Some(feature.to_string())
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    if !target_features.is_empty() {
        #[cfg(feature="master")]
        func.add_attribute(FnAttribute::Target(&target_features));
    }
}
