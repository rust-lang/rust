//! Error Reporting for `impl` items that do not match the obligations from their `trait`.

use crate::hir;
use crate::hir::def_id::DefId;
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::{Subtype, ValuePairs};
use crate::traits::ObligationCauseCode::CompareImplMethodObligation;
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::ErrorReported;
use rustc_middle::ty::error::ExpectedFound;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::{self, Ty};
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
        err.span_label(sp, &format!("found {:?}", found));
        err.span_label(trait_sp, &format!("expected {:?}", expected));
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
                debug!("assoc type finder ty {:?} {:?}", ty, ty.kind);
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
            if let hir::ItemKind::Trait(_, _, generics, _, _) = &trait_item.kind {
                for param_ty in visitor.0 {
                    if let Some(generic) = generics.get_named(param_ty.name) {
                        err.span_label(generic.span, &format!(
                            "for `impl` items to implement the method, this type parameter might \
                             need a lifetime restriction like `{}: 'a`",
                            param_ty.name,
                        ));
                    }
                }
            }
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
