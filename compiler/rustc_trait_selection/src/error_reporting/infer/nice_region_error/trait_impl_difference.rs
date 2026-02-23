//! Error Reporting for `impl` items that do not match the obligations from their `trait`.

use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::{Namespace, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{Visitor, walk_ty};
use rustc_hir::{self as hir, AmbigArg};
use rustc_infer::infer::SubregionOrigin;
use rustc_middle::hir::nested_filter;
use rustc_middle::traits::ObligationCauseCode;
use rustc_middle::ty::error::ExpectedFound;
use rustc_middle::ty::print::RegionHighlightMode;
use rustc_middle::ty::{self, TyCtxt, TypeVisitable};
use rustc_span::{Ident, Span};
use tracing::debug;

use crate::error_reporting::infer::nice_region_error::NiceRegionError;
use crate::error_reporting::infer::nice_region_error::placeholder_error::Highlighted;
use crate::errors::{ConsiderBorrowingParamHelp, TraitImplDiff};
use crate::infer::{RegionResolutionError, ValuePairs};

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
            && let (SubregionOrigin::Subtype(sup_trace), SubregionOrigin::Subtype(sub_trace)) =
                (&sup_origin, &sub_origin)
            && let &ObligationCauseCode::CompareImplItem { trait_item_def_id, .. } =
                sub_trace.cause.code()
            && sub_trace.values == sup_trace.values
            && let ValuePairs::PolySigs(ExpectedFound { expected, found }) = sub_trace.values
        {
            // FIXME(compiler-errors): Don't like that this needs `Ty`s, but
            // all of the region highlighting machinery only deals with those.
            let guar = self.emit_err(var_origin.span(), expected, found, trait_item_def_id);
            return Some(guar);
        }
        None
    }

    fn emit_err(
        &self,
        sp: Span,
        expected: ty::PolyFnSig<'tcx>,
        found: ty::PolyFnSig<'tcx>,
        trait_item_def_id: DefId,
    ) -> ErrorGuaranteed {
        let trait_sp = self.tcx().def_span(trait_item_def_id);

        // Mark all unnamed regions in the type with a number.
        // This diagnostic is called in response to lifetime errors, so be informative.
        struct HighlightBuilder<'tcx> {
            tcx: TyCtxt<'tcx>,
            highlight: RegionHighlightMode<'tcx>,
            counter: usize,
        }

        impl<'tcx> HighlightBuilder<'tcx> {
            fn build(tcx: TyCtxt<'tcx>, sig: ty::PolyFnSig<'tcx>) -> RegionHighlightMode<'tcx> {
                let mut builder =
                    HighlightBuilder { tcx, highlight: RegionHighlightMode::default(), counter: 1 };
                sig.visit_with(&mut builder);
                builder.highlight
            }
        }

        impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for HighlightBuilder<'tcx> {
            fn visit_region(&mut self, r: ty::Region<'tcx>) {
                if !r.is_named(self.tcx) && self.counter <= 3 {
                    self.highlight.highlighting_region(r, self.counter);
                    self.counter += 1;
                }
            }
        }

        let tcx = self.cx.tcx;
        let expected_highlight = HighlightBuilder::build(tcx, expected);
        let expected = Highlighted {
            highlight: expected_highlight,
            ns: Namespace::TypeNS,
            tcx,
            value: expected,
        }
        .to_string();
        let found_highlight = HighlightBuilder::build(tcx, found);
        let found =
            Highlighted { highlight: found_highlight, ns: Namespace::TypeNS, tcx, value: found }
                .to_string();

        // Get the span of all the used type parameters in the method.
        let assoc_item = self.tcx().associated_item(trait_item_def_id);
        let mut visitor =
            TypeParamSpanVisitor { tcx: self.tcx(), types: vec![], elided_lifetime_paths: vec![] };
        match assoc_item.kind {
            ty::AssocKind::Fn { .. } => {
                if let Some(hir_id) =
                    assoc_item.def_id.as_local().map(|id| self.tcx().local_def_id_to_hir_id(id))
                    && let Some(decl) = self.tcx().hir_fn_decl_by_hir_id(hir_id)
                {
                    visitor.visit_fn_decl(decl);
                }
            }
            _ => {}
        }

        let diag = TraitImplDiff {
            sp,
            trait_sp,
            note: (),
            param_help: ConsiderBorrowingParamHelp { spans: visitor.types.to_vec() },
            rel_help: visitor.types.is_empty(),
            expected,
            found,
        };

        let mut diag = self.tcx().dcx().create_err(diag);
        // A limit not to make diag verbose.
        const ELIDED_LIFETIME_NOTE_LIMIT: usize = 5;
        let elided_lifetime_paths = visitor.elided_lifetime_paths;
        let total_elided_lifetime_paths = elided_lifetime_paths.len();
        let shown_elided_lifetime_paths = if tcx.sess.opts.verbose {
            total_elided_lifetime_paths
        } else {
            ELIDED_LIFETIME_NOTE_LIMIT
        };

        for elided in elided_lifetime_paths.into_iter().take(shown_elided_lifetime_paths) {
            diag.span_note(
                elided.span,
                format!("`{}` here is elided as `{}`", elided.ident, elided.shorthand),
            );
        }
        if total_elided_lifetime_paths > shown_elided_lifetime_paths {
            diag.note(format!(
                "and {} more elided lifetime{} in type paths",
                total_elided_lifetime_paths - shown_elided_lifetime_paths,
                if total_elided_lifetime_paths - shown_elided_lifetime_paths == 1 {
                    ""
                } else {
                    "s"
                },
            ));
        }
        diag.emit()
    }
}

#[derive(Clone)]
struct ElidedLifetimeInPath {
    span: Span,
    ident: Ident,
    shorthand: String,
}

struct TypeParamSpanVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    types: Vec<Span>,
    elided_lifetime_paths: Vec<ElidedLifetimeInPath>,
}

impl<'tcx> Visitor<'tcx> for TypeParamSpanVisitor<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_qpath(&mut self, qpath: &'tcx hir::QPath<'tcx>, id: hir::HirId, _span: Span) {
        fn record_elided_lifetimes(
            tcx: TyCtxt<'_>,
            elided_lifetime_paths: &mut Vec<ElidedLifetimeInPath>,
            segment: &hir::PathSegment<'_>,
        ) {
            let Some(args) = segment.args else { return };
            if args.parenthesized != hir::GenericArgsParentheses::No {
                // Our diagnostic rendering below uses `<...>` syntax; skip cases like `Fn(..) -> ..`.
                return;
            }
            let elided_count = args
                .args
                .iter()
                .filter(|arg| {
                    let hir::GenericArg::Lifetime(l) = arg else { return false };
                    l.syntax == hir::LifetimeSyntax::Implicit
                        && matches!(l.source, hir::LifetimeSource::Path { .. })
                })
                .count();
            if elided_count == 0
                || elided_lifetime_paths.iter().any(|p| p.span == segment.ident.span)
            {
                return;
            }

            let sm = tcx.sess.source_map();
            let mut parts = args
                .args
                .iter()
                .map(|arg| match arg {
                    hir::GenericArg::Lifetime(l) => {
                        if l.syntax == hir::LifetimeSyntax::Implicit
                            && matches!(l.source, hir::LifetimeSource::Path { .. })
                        {
                            "'_".to_string()
                        } else {
                            sm.span_to_snippet(l.ident.span)
                                .unwrap_or_else(|_| format!("'{}", l.ident.name))
                        }
                    }
                    hir::GenericArg::Type(ty) => {
                        sm.span_to_snippet(ty.span).unwrap_or_else(|_| "..".to_string())
                    }
                    hir::GenericArg::Const(ct) => {
                        sm.span_to_snippet(ct.span).unwrap_or_else(|_| "..".to_string())
                    }
                    hir::GenericArg::Infer(_) => "_".to_string(),
                })
                .collect::<Vec<_>>();
            parts.extend(args.constraints.iter().map(|constraint| {
                sm.span_to_snippet(constraint.span)
                    .unwrap_or_else(|_| format!("{} = ..", constraint.ident))
            }));
            let shorthand = format!("{}<{}>", segment.ident, parts.join(", "));

            elided_lifetime_paths.push(ElidedLifetimeInPath {
                span: segment.ident.span,
                ident: segment.ident,
                shorthand,
            });
        }

        match qpath {
            hir::QPath::Resolved(_, path) => {
                for segment in path.segments {
                    record_elided_lifetimes(self.tcx, &mut self.elided_lifetime_paths, segment);
                }
            }
            hir::QPath::TypeRelative(_, segment) => {
                record_elided_lifetimes(self.tcx, &mut self.elided_lifetime_paths, segment);
            }
        }

        hir::intravisit::walk_qpath(self, qpath, id);
    }

    fn visit_ty(&mut self, arg: &'tcx hir::Ty<'tcx, AmbigArg>) {
        match arg.kind {
            hir::TyKind::Ref(_, ref mut_ty) => {
                // We don't want to suggest looking into borrowing `&T` or `&Self`.
                if let Some(ambig_ty) = mut_ty.ty.try_as_ambig_ty() {
                    walk_ty(self, ambig_ty);
                }
                return;
            }
            hir::TyKind::Path(hir::QPath::Resolved(None, path)) => match &path.segments {
                [segment]
                    if matches!(
                        segment.res,
                        Res::SelfTyParam { .. }
                            | Res::SelfTyAlias { .. }
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
