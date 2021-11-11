//! Error Reporting for `impl` items that do not match the obligations from their `trait`.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::{SubregionOrigin, Subtype, ValuePairs};
use crate::traits::ObligationCauseCode::CompareImplMethodObligation;
use rustc_errors::ErrorReported;
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::Visitor;
use rustc_middle::ty::print::RegionHighlightMode;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable, TypeVisitor};

use rustc_span::{MultiSpan, Span, Symbol};

use std::ops::ControlFlow;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// Print the error message for lifetime errors when the `impl` doesn't conform to the `trait`.
    pub(super) fn try_report_impl_not_conforming_to_trait(&self) -> Option<ErrorReported> {
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
        {
            if let (&Subtype(ref sup_trace), &Subtype(ref sub_trace)) = (&sup_origin, &sub_origin) {
                if let (
                    ValuePairs::Types(sub_expected_found),
                    ValuePairs::Types(sup_expected_found),
                    CompareImplMethodObligation { trait_item_def_id, .. },
                ) = (&sub_trace.values, &sup_trace.values, sub_trace.cause.code())
                {
                    if sup_expected_found == sub_expected_found {
                        self.emit_err(
                            var_origin.span(),
                            sub_expected_found.expected,
                            sub_expected_found.found,
                            *trait_item_def_id,
                        );
                        return Some(ErrorReported);
                    }
                }
            }
        }
        if let RegionResolutionError::ConcreteFailure(origin, _, _)
        | RegionResolutionError::GenericBoundFailure(origin, _, _) = error.clone()
        {
            if let SubregionOrigin::CompareImplTypeObligation {
                span,
                impl_item_def_id,
                trait_item_def_id,
            } = origin
            {
                self.emit_associated_type_err(
                    span,
                    self.infcx.tcx.item_name(impl_item_def_id),
                    impl_item_def_id,
                    trait_item_def_id,
                );
                return Some(ErrorReported);
            }
        }
        None
    }

    fn emit_err(&self, sp: Span, expected: Ty<'tcx>, found: Ty<'tcx>, trait_def_id: DefId) {
        let trait_sp = self.tcx().def_span(trait_def_id);
        let mut err = self
            .tcx()
            .sess
            .struct_span_err(sp, "`impl` item signature doesn't match `trait` item signature");

        // Mark all unnamed regions in the type with a number.
        // This diagnostic is called in response to lifetime errors, so be informative.
        struct HighlightBuilder<'tcx> {
            highlight: RegionHighlightMode,
            tcx: TyCtxt<'tcx>,
            counter: usize,
        }

        impl<'tcx> HighlightBuilder<'tcx> {
            fn build(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> RegionHighlightMode {
                let mut builder =
                    HighlightBuilder { highlight: RegionHighlightMode::default(), counter: 1, tcx };
                builder.visit_ty(ty);
                builder.highlight
            }
        }

        impl<'tcx> ty::fold::TypeVisitor<'tcx> for HighlightBuilder<'tcx> {
            fn tcx_for_anon_const_substs(&self) -> Option<TyCtxt<'tcx>> {
                Some(self.tcx)
            }

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
            type_param_span.push_span_label(
                span,
                "consider borrowing this type parameter in the trait".to_string(),
            );
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
        err.emit();
    }

    fn emit_associated_type_err(
        &self,
        span: Span,
        item_name: Symbol,
        impl_item_def_id: DefId,
        trait_item_def_id: DefId,
    ) {
        let impl_sp = self.tcx().def_span(impl_item_def_id);
        let trait_sp = self.tcx().def_span(trait_item_def_id);
        let mut err = self
            .tcx()
            .sess
            .struct_span_err(span, &format!("`impl` associated type signature for `{}` doesn't match `trait` associated type signature", item_name));
        err.span_label(impl_sp, "found");
        err.span_label(trait_sp, "expected");

        err.emit();
    }
}

struct TypeParamSpanVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    types: Vec<Span>,
}

impl<'tcx> Visitor<'tcx> for TypeParamSpanVisitor<'tcx> {
    type Map = rustc_middle::hir::map::Map<'tcx>;

    fn nested_visit_map(&mut self) -> hir::intravisit::NestedVisitorMap<Self::Map> {
        hir::intravisit::NestedVisitorMap::OnlyBodies(self.tcx.hir())
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
                    if segment
                        .res
                        .map(|res| {
                            matches!(
                                res,
                                Res::SelfTy(_, _) | Res::Def(hir::def::DefKind::TyParam, _)
                            )
                        })
                        .unwrap_or(false) =>
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
