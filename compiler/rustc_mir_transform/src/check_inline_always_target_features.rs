use rustc_hir::attrs::InlineAttr;
use rustc_middle::middle::codegen_fn_attrs::TargetFeatureKind;
use rustc_middle::mir::{Body, TerminatorKind};
use rustc_middle::ty::{self, TyCtxt};

use crate::pass_manager::MirLint;

pub(super) struct CheckInlineAlwaysTargetFeature;

impl<'tcx> MirLint<'tcx> for CheckInlineAlwaysTargetFeature {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        check_inline_always_target_features(tcx, body)
    }
}

/// `#[target_feature]`-annotated functions can be marked `#[inline]` and will only be inlined if
/// the target features match (as well as all of the other inlining heuristics). `#[inline(always)]`
/// will always inline regardless of matching target features, which can result in errors from LLVM.
/// However, it is desirable to be able to always annotate certain functions (e.g. SIMD intrinsics)
/// as `#[inline(always)]` but check the target features match in Rust to avoid the LLVM errors.
///
/// We check the caller and callee target features to ensure that this can
/// be done or emit a lint.
#[inline]
fn check_inline_always_target_features<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
    let caller_def_id = body.source.def_id().expect_local();
    if !tcx.def_kind(caller_def_id).has_codegen_attrs() {
        return;
    }

    let caller_codegen_fn_attrs = tcx.codegen_fn_attrs(caller_def_id);

    for bb in body.basic_blocks.iter() {
        let terminator = bb.terminator();
        match &terminator.kind {
            TerminatorKind::Call { func, .. } | TerminatorKind::TailCall { func, .. } => {
                let fn_ty = func.ty(body, tcx);
                let ty::FnDef(callee_def_id, _) = *fn_ty.kind() else {
                    continue;
                };

                if !tcx.def_kind(callee_def_id).has_codegen_attrs() {
                    continue;
                }
                let callee_codegen_fn_attrs = tcx.codegen_fn_attrs(callee_def_id);
                if callee_codegen_fn_attrs.inline != InlineAttr::Always
                    || callee_codegen_fn_attrs.target_features.is_empty()
                {
                    continue;
                }

                // Scan the users defined target features and ensure they
                // match the caller.
                if tcx.is_call_inline_able_at_callsite(
                    &callee_codegen_fn_attrs.target_features,
                    &caller_codegen_fn_attrs.target_features,
                ) {
                    continue;
                }

                // Use the full target feature sets, including implied and
                // command-line features, to classify the mismatch. Diagnostic
                // messages should still only mention the non-implied features
                // that the user actually enabled.
                let caller_features =
                    tcx.effective_inline_target_features(&caller_codegen_fn_attrs.target_features);
                let callee_features =
                    tcx.effective_inline_target_features(&callee_codegen_fn_attrs.target_features);

                let explicit_caller_features: Vec<_> = caller_codegen_fn_attrs
                    .target_features
                    .iter()
                    .cloned()
                    .filter(|it| it.kind != TargetFeatureKind::Implied)
                    .collect();
                let explicit_callee_features: Vec<_> = callee_codegen_fn_attrs
                    .target_features
                    .iter()
                    .cloned()
                    .filter(|it| it.kind != TargetFeatureKind::Implied)
                    .collect();

                let explicit_caller_features =
                    tcx.effective_inline_target_features(&explicit_caller_features);
                let explicit_callee_features =
                    tcx.effective_inline_target_features(&explicit_callee_features);

                // If the callee's features are otherwise a subset of the
                // caller's, then the mismatch is only due to the caller using a
                // different vector ABI from the callee.
                if callee_features.is_subset(&caller_features) {
                    // We only want to display the target features the user
                    // missed out. Not every feature that is possibly enabled.
                    let caller_abi_features = tcx.abi_target_features(&explicit_caller_features);
                    let callee_abi_features = tcx.abi_target_features(&explicit_callee_features);
                    let caller_only = caller_abi_features
                        .difference(&callee_abi_features)
                        .map(|it| it.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");

                    // Emit that the issue is caused by a vector ABI mismatch.
                    crate::errors::emit_inline_always_target_feature_diagnostic(
                        tcx,
                        terminator.source_info.span,
                        callee_def_id,
                        caller_def_id.into(),
                        &caller_only,
                        caller_def_id.into(),
                        callee_def_id,
                    );
                } else {
                    let callee_only = explicit_callee_features
                        .difference(&explicit_caller_features)
                        .map(|it| it.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");

                    // Emit that the issue stems from the callee having features
                    // enabled that the caller does not have enabled.
                    crate::errors::emit_inline_always_target_feature_diagnostic(
                        tcx,
                        terminator.source_info.span,
                        callee_def_id,
                        caller_def_id.into(),
                        &callee_only,
                        callee_def_id,
                        caller_def_id.into(),
                    );
                }
            }
            _ => (),
        }
    }
}
