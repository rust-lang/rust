//! Error Reporting for `impl` items that do not match the obligations from their `trait`.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::{Subtype, TyCtxtInferExt, ValuePairs};
use crate::traits::ObligationCauseCode::CompareImplMethodObligation;
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::ErrorReported;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::Visitor;
use rustc_hir::ItemKind;
use rustc_middle::ty::error::ExpectedFound;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// Print the error message for lifetime errors when the `impl` doesn't conform to the `trait`.
    pub(super) fn try_report_impl_not_conforming_to_trait(&self) -> Option<ErrorReported> {
        if let Some(ref error) = self.error {
            debug!("try_report_impl_not_conforming_to_trait {:?}", error);
            if let RegionResolutionError::SubSupConflict(
                _,
                var_origin,
                sub_origin,
                _sub,
                sup_origin,
                _sup,
            ) = error.clone()
            {
                if let (&Subtype(ref sup_trace), &Subtype(ref sub_trace)) =
                    (&sup_origin, &sub_origin)
                {
                    if let (
                        ValuePairs::Types(sub_expected_found),
                        ValuePairs::Types(sup_expected_found),
                        CompareImplMethodObligation { trait_item_def_id, .. },
                    ) = (&sub_trace.values, &sup_trace.values, &sub_trace.cause.code)
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
        }
        None
    }

    fn emit_err(&self, sp: Span, expected: Ty<'tcx>, found: Ty<'tcx>, trait_def_id: DefId) {
        let tcx = self.tcx();
        let trait_sp = self.tcx().def_span(trait_def_id);
        let mut err = self
            .tcx()
            .sess
            .struct_span_err(sp, "`impl` item signature doesn't match `trait` item signature");
        err.span_label(sp, &format!("found `{:?}`", found));
        err.span_label(trait_sp, &format!("expected `{:?}`", expected));
        let trait_fn_sig = tcx.fn_sig(trait_def_id);

        // Check the `trait`'s method's output to look for type parameters that might have
        // unconstrained lifetimes. If the method returns a type parameter and the `impl` has a
        // borrow as the type parameter being implemented, the lifetimes will not match because
        // a new lifetime is being introduced in the `impl` that is not present in the `trait`.
        // Because this is confusing as hell the first time you see it, we give a short message
        // explaining the situation and proposing constraining the type param with a named lifetime
        // so that the `impl` will have one to tie them together.
        struct AssocTypeFinder(FxIndexSet<ty::ParamTy>);
        impl<'tcx> ty::fold::TypeVisitor<'tcx> for AssocTypeFinder {
            fn visit_ty(&mut self, ty: Ty<'tcx>) -> bool {
                if let ty::Param(param) = ty.kind {
                    self.0.insert(param);
                }
                ty.super_visit_with(self)
            }
        }
        let mut visitor = AssocTypeFinder(FxIndexSet::default());
        trait_fn_sig.output().visit_with(&mut visitor);
        if let Some(id) = tcx.hir().as_local_hir_id(trait_def_id) {
            let parent_id = tcx.hir().get_parent_item(id);
            let trait_item = tcx.hir().expect_item(parent_id);
            if let ItemKind::Trait(_, _, generics, _, _) = &trait_item.kind {
                for param_ty in &visitor.0 {
                    if let Some(generic) = generics.get_named(param_ty.name) {
                        err.span_label(
                            generic.span,
                            "this type parameter might not have a lifetime compatible with the \
                             `impl`",
                        );
                    }
                }
            }
        }

        // Get the span of all the used type parameters in the method.
        let assoc_item = self.tcx().associated_item(trait_def_id);
        let mut visitor = TypeParamSpanVisitor { tcx: self.tcx(), types: vec![] };
        match assoc_item.kind {
            ty::AssocKind::Method => {
                let hir = self.tcx().hir();
                if let Some(hir_id) = hir.as_local_hir_id(assoc_item.def_id) {
                    if let Some(decl) = hir.fn_decl_by_hir_id(hir_id) {
                        visitor.visit_fn_decl(decl);
                    }
                }
            }
            _ => {}
        }
        for span in visitor.types {
            err.span_label(
                span,
                "you might want to borrow this type parameter in the trait to make it match the \
                 `impl`",
            );
        }

        if let Some((expected, found)) = tcx
            .infer_ctxt()
            .enter(|infcx| infcx.expected_found_str_ty(&ExpectedFound { expected, found }))
        {
            // Highlighted the differences when showing the "expected/found" note.
            err.note_expected_found(&"", expected, &"", found);
        } else {
            // This fallback shouldn't be necessary, but let's keep it in just in case.
            err.note(&format!("expected `{:?}`\n   found `{:?}`", expected, found));
        }
        err.note("the lifetime requirements from the `trait` could not be satisfied by the `impl`");
        err.help(
            "verify the lifetime relationships in the `trait` and `impl` between the `self` \
             argument, the other inputs and its output",
        );
        err.emit();
    }
}

struct TypeParamSpanVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    types: Vec<Span>,
}

impl Visitor<'tcx> for TypeParamSpanVisitor<'tcx> {
    type Map = hir::intravisit::Map<'tcx>;

    fn nested_visit_map(&mut self) -> hir::intravisit::NestedVisitorMap<Self::Map> {
        hir::intravisit::NestedVisitorMap::OnlyBodies(self.tcx.hir())
    }

    fn visit_ty(&mut self, arg: &'tcx hir::Ty<'tcx>) {
        match arg.kind {
            hir::TyKind::Slice(_) | hir::TyKind::Tup(_) | hir::TyKind::Array(..) => {
                hir::intravisit::walk_ty(self, arg);
            }
            hir::TyKind::Path(hir::QPath::Resolved(None, path)) => match &path.segments {
                [segment]
                    if segment
                        .res
                        .map(|res| match res {
                            hir::def::Res::Def(hir::def::DefKind::TyParam, _) => true,
                            _ => false,
                        })
                        .unwrap_or(false) =>
                {
                    self.types.push(path.span);
                }
                _ => {}
            },
            _ => {}
        }
    }
}
