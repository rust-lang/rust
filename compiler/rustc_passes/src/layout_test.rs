use rustc_abi::{HasDataLayout, TargetDataLayout};
use rustc_hir::attrs::RustcDumpLayoutKind;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::find_attr;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv, LayoutError, LayoutOfHelpers};
use rustc_middle::ty::{self, Ty, TyCtxt, Unnormalized};
use rustc_span::Span;
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::infer::TyCtxtInferExt;
use rustc_trait_selection::traits;

pub fn test_layout(tcx: TyCtxt<'_>) {
    if !tcx.features().rustc_attrs() {
        // if the `rustc_attrs` feature is not enabled, don't bother testing layout
        return;
    }
    for id in tcx.hir_crate_items(()).definitions() {
        if let Some(kinds) = find_attr!(tcx, id, RustcDumpLayout(kinds) => kinds) {
            // Attribute parsing handles error reporting
            if let DefKind::TyAlias | DefKind::Enum | DefKind::Struct | DefKind::Union =
                tcx.def_kind(id)
            {
                dump_layout_of(tcx, id, kinds);
            }
        }
    }
}

pub fn ensure_wf<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
    def_id: LocalDefId,
    span: Span,
) -> bool {
    let (infcx, param_env) = tcx.infer_ctxt().build_with_typing_env(typing_env);
    let ocx = traits::ObligationCtxt::new_with_diagnostics(&infcx);
    let pred = ty::ClauseKind::WellFormed(ty.into());
    let obligation = traits::Obligation::new(
        tcx,
        traits::ObligationCause::new(
            span,
            def_id,
            traits::ObligationCauseCode::WellFormed(Some(traits::WellFormedLoc::Ty(def_id))),
        ),
        param_env,
        pred,
    );
    ocx.register_obligation(obligation);
    let errors = ocx.evaluate_obligations_error_on_ambiguity();
    if !errors.is_empty() {
        infcx.err_ctxt().report_fulfillment_errors(errors);
        false
    } else {
        // looks WF!
        true
    }
}

fn dump_layout_of(tcx: TyCtxt<'_>, item_def_id: LocalDefId, kinds: &[RustcDumpLayoutKind]) {
    let typing_env = ty::TypingEnv::post_analysis(tcx, item_def_id);
    let ty = tcx.type_of(item_def_id).instantiate_identity().skip_norm_wip();
    let span = tcx.def_span(item_def_id.to_def_id());
    if !ensure_wf(tcx, typing_env, ty, item_def_id, span) {
        return;
    }
    match tcx.layout_of(typing_env.as_query_input(ty)) {
        Ok(ty_layout) => {
            for kind in kinds {
                let message = match kind {
                    RustcDumpLayoutKind::Align => format!("align: {:?}", *ty_layout.align),
                    RustcDumpLayoutKind::BackendRepr => {
                        format!("backend_repr: {:?}", ty_layout.backend_repr)
                    }
                    RustcDumpLayoutKind::Debug => {
                        let normalized_ty =
                            tcx.normalize_erasing_regions(typing_env, Unnormalized::new_wip(ty));
                        // FIXME: using the `Debug` impl here isn't ideal.
                        format!("layout_of({normalized_ty}) = {:#?}", *ty_layout)
                    }
                    RustcDumpLayoutKind::HomogenousAggregate => {
                        let data =
                            ty_layout.homogeneous_aggregate(&UnwrapLayoutCx { tcx, typing_env });
                        format!("homogeneous_aggregate: {data:?}")
                    }
                    RustcDumpLayoutKind::Size => format!("size: {:?}", ty_layout.size),
                };
                tcx.dcx().span_err(span, message);
            }
        }

        Err(layout_error) => {
            tcx.dcx().span_err(span, layout_error.to_string());
        }
    }
}

struct UnwrapLayoutCx<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
}

impl<'tcx> LayoutOfHelpers<'tcx> for UnwrapLayoutCx<'tcx> {
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        span_bug!(
            span,
            "`#[rustc_dump_layout(..)]` test resulted in `layout_of({ty}) = Err({err})`",
        );
    }
}

impl<'tcx> HasTyCtxt<'tcx> for UnwrapLayoutCx<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> HasTypingEnv<'tcx> for UnwrapLayoutCx<'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.typing_env
    }
}

impl<'tcx> HasDataLayout for UnwrapLayoutCx<'tcx> {
    fn data_layout(&self) -> &TargetDataLayout {
        self.tcx.data_layout()
    }
}
