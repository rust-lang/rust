use rustc_abi::{HasDataLayout, TargetDataLayout};
use rustc_hir::attrs::{AttributeKind, RustcLayoutType};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::find_attr;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv, LayoutError, LayoutOfHelpers};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;
use rustc_span::source_map::Spanned;
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::infer::TyCtxtInferExt;
use rustc_trait_selection::traits;

use crate::errors::{LayoutAbi, LayoutAlign, LayoutHomogeneousAggregate, LayoutOf, LayoutSize};

pub fn test_layout(tcx: TyCtxt<'_>) {
    if !tcx.features().rustc_attrs() {
        // if the `rustc_attrs` feature is not enabled, don't bother testing layout
        return;
    }
    for id in tcx.hir_crate_items(()).definitions() {
        let attrs = tcx.get_all_attrs(id);
        if let Some(attrs) = find_attr!(attrs, AttributeKind::RustcLayout(attrs) => attrs) {
            // Attribute parsing handles error reporting
            if matches!(
                tcx.def_kind(id),
                DefKind::TyAlias | DefKind::Enum | DefKind::Struct | DefKind::Union
            ) {
                dump_layout_of(tcx, id, attrs);
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

fn dump_layout_of(tcx: TyCtxt<'_>, item_def_id: LocalDefId, attrs: &[RustcLayoutType]) {
    let typing_env = ty::TypingEnv::post_analysis(tcx, item_def_id);
    let ty = tcx.type_of(item_def_id).instantiate_identity();
    let span = tcx.def_span(item_def_id.to_def_id());
    if !ensure_wf(tcx, typing_env, ty, item_def_id, span) {
        return;
    }
    match tcx.layout_of(typing_env.as_query_input(ty)) {
        Ok(ty_layout) => {
            for attr in attrs {
                match attr {
                    // FIXME: this never was about ABI and now this dump arg is confusing
                    RustcLayoutType::Abi => {
                        tcx.dcx().emit_err(LayoutAbi {
                            span,
                            abi: format!("{:?}", ty_layout.backend_repr),
                        });
                    }

                    RustcLayoutType::Align => {
                        tcx.dcx().emit_err(LayoutAlign {
                            span,
                            align: format!("{:?}", ty_layout.align),
                        });
                    }

                    RustcLayoutType::Size => {
                        tcx.dcx()
                            .emit_err(LayoutSize { span, size: format!("{:?}", ty_layout.size) });
                    }

                    RustcLayoutType::HomogenousAggregate => {
                        tcx.dcx().emit_err(LayoutHomogeneousAggregate {
                            span,
                            homogeneous_aggregate: format!(
                                "{:?}",
                                ty_layout
                                    .homogeneous_aggregate(&UnwrapLayoutCx { tcx, typing_env })
                            ),
                        });
                    }

                    RustcLayoutType::Debug => {
                        let normalized_ty = tcx.normalize_erasing_regions(typing_env, ty);
                        // FIXME: using the `Debug` impl here isn't ideal.
                        let ty_layout = format!("{:#?}", *ty_layout);
                        tcx.dcx().emit_err(LayoutOf { span, normalized_ty, ty_layout });
                    }
                }
            }
        }

        Err(layout_error) => {
            tcx.dcx().emit_err(Spanned { node: layout_error.into_diagnostic(), span });
        }
    }
}

struct UnwrapLayoutCx<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
}

impl<'tcx> LayoutOfHelpers<'tcx> for UnwrapLayoutCx<'tcx> {
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        span_bug!(span, "`#[rustc_layout(..)]` test resulted in `layout_of({ty}) = Err({err})`",);
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
