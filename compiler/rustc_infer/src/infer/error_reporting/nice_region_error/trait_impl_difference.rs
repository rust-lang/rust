//! Error Reporting for `impl` items that do not match the obligations from their `trait`.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::Subtype;
use crate::traits::ObligationCauseCode::CompareImplItemObligation;
use rustc_errors::{ErrorGuaranteed, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::Visitor;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::print::RegionHighlightMode;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitor};
use rustc_span::Span;

use std::ops::ControlFlow;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// Print the error message for lifetime errors when the `impl` doesn't conform to the `trait`.
    pub(super) fn try_report_impl_not_conforming_to_trait(&self) -> Option<ErrorGuaranteed> {
        let error = self.error.as_ref()?;
        debug!("try_report_impl_not_conforming_to_trait {:?}", error);
        if let RegionResolutionError::SubSupConflict(
                _,
                var_origin,
                sub_origin,
                _sub,
                sup_origin,
                _sup,
                _,
            ) = error.clone()
            && let (Subtype(sup_trace), Subtype(sub_trace)) = (&sup_origin, &sub_origin)
            && let sub_expected_found @ Some((sub_expected, sub_found)) = sub_trace.values.ty()
            && let sup_expected_found @ Some(_) = sup_trace.values.ty()
            && let CompareImplItemObligation { trait_item_def_id, .. } = sub_trace.cause.code()
            && sup_expected_found == sub_expected_found
        {
            let guar =
                self.emit_err(var_origin.span(), sub_expected, sub_found, *trait_item_def_id);
            return Some(guar);
        }
        None
    }

    fn emit_err(
        &self,
        sp: Span,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        trait_def_id: DefId,
    ) -> ErrorGuaranteed {
        let trait_sp = self.tcx().def_span(trait_def_id);
        let mut err = self
            .tcx()
            .sess
            .struct_span_err(sp, "`impl` item signature doesn't match `trait` item signature");

        // Mark all unnamed regions in the type with a number.
        // This diagnostic is called in response to lifetime errors, so be informative.
        struct HighlightBuilder<'tcx> {
            highlight: RegionHighlightMode<'tcx>,
            counter: usize,
        }

        impl<'tcx> HighlightBuilder<'tcx> {
            fn build(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> RegionHighlightMode<'tcx> {
                let mut builder =
                    HighlightBuilder { highlight: RegionHighlightMode::new(tcx), counter: 1 };
                builder.visit_ty(ty);
                builder.highlight
            }
        }

        impl<'tcx> ty::visit::TypeVisitor<'tcx> for HighlightBuilder<'tcx> {
            fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
                if !r.has_name() && self.counter <= 3 {
                    self.highlight.highlighting_region(r, self.counter);
                    self.counter += 1;
                }
                r.super_visit_with(self)
            }
        }

        let expected_highlight = HighlightBuilder::build(self.tcx(), expected);
        let expected = self
            .infcx
            .extract_inference_diagnostics_data(expected.into(), Some(expected_highlight))
            .name;
        let found_highlight = HighlightBuilder::build(self.tcx(), found);
        let found =
            self.infcx.extract_inference_diagnostics_data(found.into(), Some(found_highlight)).name;

        err.span_label(sp, &format!("found `{}`", found));
        err.span_label(trait_sp, &format!("expected `{}`", expected));

        // Get the span of all the used type parameters in the method.
        let assoc_item = self.tcx().associated_item(trait_def_id);
        let mut visitor = TypeParamSpanVisitor { tcx: self.tcx(), types: vec![] };
        match assoc_item.kind {
            ty::AssocKind::Fn => {
                let hir = self.tcx().hir();
                if let Some(hir_id) =
                    assoc_item.def_id.as_local().map(|id| hir.local_def_id_to_hir_id(id))
                {
                    if let Some(decl) = hir.fn_decl_by_hir_id(hir_id) {
                        visitor.visit_fn_decl(decl);
                    }
                }
            }
            _ => {}
        }
        let mut type_param_span: MultiSpan = visitor.types.to_vec().into();
        for &span in &visitor.types {
            type_param_span
                .push_span_label(span, "consider borrowing this type parameter in the trait");
        }

        err.note(&format!("expected `{}`\n   found `{}`", expected, found));

        err.span_help(
            type_param_span,
            "the lifetime requirements from the `impl` do not correspond to the requirements in \
             the `trait`",
        );
        if visitor.types.is_empty() {
            err.help(
                "verify the lifetime relationships in the `trait` and `impl` between the `self` \
                 argument, the other inputs and its output",
            );
        }
        err.emit()
    }
}

struct TypeParamSpanVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    types: Vec<Span>,
}

impl<'tcx> Visitor<'tcx> for TypeParamSpanVisitor<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_ty(&mut self, arg: &'tcx hir::Ty<'tcx>) {
        match arg.kind {
            hir::TyKind::Rptr(_, ref mut_ty) => {
                // We don't want to suggest looking into borrowing `&T` or `&Self`.
                hir::intravisit::walk_ty(self, mut_ty.ty);
                return;
            }
            hir::TyKind::Path(hir::QPath::Resolved(None, path)) => match &path.segments {
                [segment]
                    if matches!(
                        segment.res,
                        Res::SelfTy { trait_: _, alias_to: _ }
                            | Res::Def(hir::def::DefKind::TyParam, _)
                    ) =>
                {
                    self.types.push(path.span);
                }
                _ => {}
            },
            _ => {}
        }
        hir::intravisit::walk_ty(self, arg);
    }
}
