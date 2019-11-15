#![allow(unused)]
//! Error Reporting for dyn Traits.
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::ty::{self, BoundRegion, FreeRegion, RegionKind, DefIdTree, ParamEnv};
use crate::util::common::ErrorReported;
use errors::Applicability;
use crate::infer::{SubregionOrigin, ValuePairs, TypeTrace};
use crate::ty::error::ExpectedFound;
use crate::hir;
use crate::hir::def_id::DefId;
use crate::traits::ObligationCauseCode::ExprAssignable;
use crate::traits::ObligationCause;
use crate::ty::{TyCtxt, TypeFoldable};
use crate::ty::subst::{Subst, InternalSubsts, SubstsRef};
use syntax_pos::Span;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// Print the error message for lifetime errors when a static ref to a trait object is required
    /// because of a `dyn Trait` impl.
    pub(super) fn try_report_static_dyn_trait(&self) -> Option<ErrorReported> {
        let (span, sub, sup) = self.regions();

        debug!(
            "try_report_static_dyn_trait: sub={:?}, sup={:?}, error={:?}",
            sub,
            sup,
            self.error,
        );

        if let Some(ref error) = self.error {
            let (found, origin_span) = match error {
                RegionResolutionError::ConcreteFailure(SubregionOrigin::Subtype(box TypeTrace {
                    cause,
                    values: ValuePairs::TraitRefs(ExpectedFound { expected, found }),
                }), _, _) => {
                    (found, cause.span)
                },
                // FIXME there is also the other region origin!
                RegionResolutionError::SubSupConflict(_, _, SubregionOrigin::Subtype(box TypeTrace {
                    cause,
                    values: ValuePairs::TraitRefs(ExpectedFound { expected, found }),
                }), _, _, _) => {
                    (found, cause.span)
                }
                _ => {
                    return None;
                }
            };

            debug!(
                "try_report_static_dyn_trait: found={:?} origin_span={:?}",
                found, origin_span,
            );

            let found = self.infcx.resolve_vars_if_possible(found);
            let dyn_static_ty = found.self_ty().walk().find(|ty|
                if let ty::Dynamic(_, _) = ty.kind { true } else { false });

            debug!(
                "try_report_static_dyn_trait: dyn_static_ty={:?} dyn_static_ty.kind={:?}",
                dyn_static_ty, dyn_static_ty.map(|dyn_static_ty| &dyn_static_ty.kind),
            );

            let dyn_trait_name = if let Some(dyn_static_ty) = dyn_static_ty {
                if let ty::Dynamic(binder, _) = dyn_static_ty.kind {
                    binder.skip_binder().to_string()
                } else {
                    return None;
                }
            } else {
                return None;
            };

            let mut trait_impl = None;

            self.tcx().for_each_relevant_impl(
                found.def_id,
                found.self_ty(),
                |impl_def_id| {
                    debug!(
                        "try_report_static_dyn_trait: for_each_relevant_impl impl_def_id={:?}",
                        impl_def_id,
                    );

                    trait_impl = Some(impl_def_id);
                });

            debug!(
                "try_report_static_dyn_trait: trait_impl={:?}", trait_impl,
            );

            if let Some(impl_def_id) = trait_impl {
                self.emit_dyn_trait_err(origin_span, &dyn_trait_name, impl_def_id);
                return Some(ErrorReported);
            }
        }
        None
    }

    fn emit_dyn_trait_err(&self,
                          expr_span: Span,
                          dyn_trait_name: &String,
                          impl_def_id: DefId,
    ) {
        let (_, sub, sup) = self.regions();

        debug!(
            "emit_dyn_trait_err: sup={:?} sub={:?} expr_span={:?} dyn_trait_name={:?}",
            sup, sub, expr_span, dyn_trait_name,
        );

        let item_span = self.tcx().sess.source_map()
            .def_span(self.tcx().def_span(impl_def_id));

        let (lifetime_description, lt_sp_opt) = self.tcx().msg_span_from_free_region(sup);

        let impl_span = item_span;

        let mut err = self.tcx().sess.struct_span_err(
            expr_span,
            "cannot infer an appropriate lifetime",
        );

        if let Some(lt_sp_opt) = lt_sp_opt {
            err.span_note(
                lt_sp_opt,
                &format!("first, the lifetime cannot outlive {}...", lifetime_description),
            );
        }

        err.span_note(expr_span,
                      &format!("but, the lifetime must be valid for the {} lifetime...", sub));


        self.infcx.note_dyn_impl_and_suggest_anon_lifetime(&mut err, impl_def_id, dyn_trait_name);

        err.emit();
    }
}
