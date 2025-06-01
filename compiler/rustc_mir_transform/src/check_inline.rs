//! Check that a body annotated with `#[rustc_force_inline]` will not fail to inline based on its
//! definition alone (irrespective of any specific caller).

use rustc_attr_data_structures::InlineAttr;
use rustc_hir::def_id::DefId;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::{Body, TerminatorKind};
use rustc_middle::ty;
use rustc_middle::ty::TyCtxt;
use rustc_span::sym;

use crate::pass_manager::MirLint;

pub(super) struct CheckForceInline;

impl<'tcx> MirLint<'tcx> for CheckForceInline {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let def_id = body.source.def_id();
        if !tcx.hir_body_owner_kind(def_id).is_fn_or_closure() || !def_id.is_local() {
            return;
        }
        let InlineAttr::Force { attr_span, .. } = tcx.codegen_fn_attrs(def_id).inline else {
            return;
        };

        if let Err(reason) =
            is_inline_valid_on_fn(tcx, def_id).and_then(|_| is_inline_valid_on_body(tcx, body))
        {
            tcx.dcx().emit_err(crate::errors::InvalidForceInline {
                attr_span,
                callee_span: tcx.def_span(def_id),
                callee: tcx.def_path_str(def_id),
                reason,
            });
        }
    }
}

pub(super) fn is_inline_valid_on_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> Result<(), &'static str> {
    let codegen_attrs = tcx.codegen_fn_attrs(def_id);
    if tcx.has_attr(def_id, sym::rustc_no_mir_inline) {
        return Err("#[rustc_no_mir_inline]");
    }

    // FIXME(#127234): Coverage instrumentation currently doesn't handle inlined
    // MIR correctly when Modified Condition/Decision Coverage is enabled.
    if tcx.sess.instrument_coverage_mcdc() {
        return Err("incompatible with MC/DC coverage");
    }

    let ty = tcx.type_of(def_id);
    if match ty.instantiate_identity().kind() {
        ty::FnDef(..) => tcx.fn_sig(def_id).instantiate_identity().c_variadic(),
        ty::Closure(_, args) => args.as_closure().sig().c_variadic(),
        _ => false,
    } {
        return Err("C variadic");
    }

    if codegen_attrs.flags.contains(CodegenFnAttrFlags::COLD) {
        return Err("cold");
    }

    // Intrinsic fallback bodies are automatically made cross-crate inlineable,
    // but at this stage we don't know whether codegen knows the intrinsic,
    // so just conservatively don't inline it. This also ensures that we do not
    // accidentally inline the body of an intrinsic that *must* be overridden.
    if tcx.has_attr(def_id, sym::rustc_intrinsic) {
        return Err("callee is an intrinsic");
    }

    Ok(())
}

pub(super) fn is_inline_valid_on_body<'tcx>(
    _: TyCtxt<'tcx>,
    body: &Body<'tcx>,
) -> Result<(), &'static str> {
    if body
        .basic_blocks
        .iter()
        .any(|bb| matches!(bb.terminator().kind, TerminatorKind::TailCall { .. }))
    {
        return Err("can't inline functions with tail calls");
    }

    Ok(())
}
