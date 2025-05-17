#[cfg(feature = "master")]
use gccjit::FnAttribute;
use gccjit::Function;
#[cfg(feature = "master")]
use rustc_attr_data_structures::InlineAttr;
use rustc_attr_data_structures::InstructionSetAttr;
#[cfg(feature = "master")]
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
#[cfg(feature = "master")]
use rustc_middle::mir::TerminatorKind;
use rustc_middle::ty;

use crate::context::CodegenCx;
use crate::gcc_util::to_gcc_features;

/// Checks if the function `instance` is recursively inline.
/// Returns `false` if a functions is guaranteed to be non-recursive, and `true` if it *might* be recursive.
#[cfg(feature = "master")]
fn resursively_inline<'gcc, 'tcx>(
    cx: &CodegenCx<'gcc, 'tcx>,
    instance: ty::Instance<'tcx>,
) -> bool {
    // No body, so we can't check if this is recursively inline, so we assume it is.
    if !cx.tcx.is_mir_available(instance.def_id()) {
        return true;
    }
    // `expect_local` ought to never fail: we should be checking a function within this codegen unit.
    let body = cx.tcx.optimized_mir(instance.def_id());
    for block in body.basic_blocks.iter() {
        let Some(ref terminator) = block.terminator else { continue };
        // I assume that the recursive-inline issue applies only to functions, and not to drops.
        // In principle, a recursive, `#[inline(always)]` drop could(?) exist, but I don't think it does.
        let TerminatorKind::Call { ref func, .. } = terminator.kind else { continue };
        let Some((def, _args)) = func.const_fn_def() else { continue };
        // Check if the called function is recursively inline.
        if matches!(
            cx.tcx.codegen_fn_attrs(def).inline,
            InlineAttr::Always | InlineAttr::Force { .. }
        ) {
            return true;
        }
    }
    false
}

/// Get GCC attribute for the provided inline heuristic, attached to `instance`.
#[cfg(feature = "master")]
#[inline]
fn inline_attr<'gcc, 'tcx>(
    cx: &CodegenCx<'gcc, 'tcx>,
    inline: InlineAttr,
    instance: ty::Instance<'tcx>,
) -> Option<FnAttribute<'gcc>> {
    match inline {
        InlineAttr::Always => {
            // We can't simply always return `always_inline` unconditionally.
            // It is *NOT A HINT* and does not work for recursive functions.
            //
            // So, it can only be applied *if*:
            // The current function does not call any functions marked `#[inline(always)]`.
            //
            // That prevents issues steming from recursive `#[inline(always)]` at a *relatively* small cost.
            // We *only* need to check all the terminators of a function marked with this attribute.
            if resursively_inline(cx, instance) {
                Some(FnAttribute::Inline)
            } else {
                Some(FnAttribute::AlwaysInline)
            }
        }
        InlineAttr::Hint => Some(FnAttribute::Inline),
        InlineAttr::Force { .. } => Some(FnAttribute::AlwaysInline),
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
    #[cfg_attr(not(feature = "master"), allow(unused_variables))] func: Function<'gcc>,
    instance: ty::Instance<'tcx>,
) {
    let codegen_fn_attrs = cx.tcx.codegen_fn_attrs(instance.def_id());

    #[cfg(feature = "master")]
    {
        let inline = if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
            InlineAttr::Never
        } else if codegen_fn_attrs.inline == InlineAttr::None
            && instance.def.requires_inline(cx.tcx)
        {
            InlineAttr::Hint
        } else {
            codegen_fn_attrs.inline
        };
        if let Some(attr) = inline_attr(cx, inline, instance) {
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

    let mut function_features = codegen_fn_attrs
        .target_features
        .iter()
        .map(|features| features.name.as_str())
        .flat_map(|feat| to_gcc_features(cx.tcx.sess, feat).into_iter())
        .chain(codegen_fn_attrs.instruction_set.iter().map(|x| match *x {
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
            // TODO(antoyo): support soft-float.
            if feature.contains("soft-float") {
                return None;
            }

            if feature.starts_with('-') {
                Some(format!("no{}", feature))
            } else if let Some(stripped) = feature.strip_prefix('+') {
                Some(stripped.to_string())
            } else {
                Some(feature.to_string())
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    if !target_features.is_empty() {
        #[cfg(feature = "master")]
        match cx.sess().target.arch.as_ref() {
            "x86" | "x86_64" | "powerpc" => {
                func.add_attribute(FnAttribute::Target(&target_features))
            }
            // The target attribute is not supported on other targets in GCC.
            _ => (),
        }
    }
}
