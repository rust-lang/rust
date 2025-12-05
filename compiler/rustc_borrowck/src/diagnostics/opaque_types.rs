#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use std::ops::ControlFlow;

use either::Either;
use itertools::Itertools as _;
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{Diag, Subdiagnostic};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::{self, ConstraintCategory, Location};
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
};
use rustc_span::Span;
use rustc_trait_selection::error_reporting::infer::region::unexpected_hidden_region_diagnostic;
use rustc_trait_selection::errors::impl_trait_overcapture_suggestion;

use crate::MirBorrowckCtxt;
use crate::borrow_set::BorrowData;
use crate::consumers::RegionInferenceContext;
use crate::region_infer::opaque_types::DeferredOpaqueTypeError;
use crate::type_check::Locations;

impl<'infcx, 'tcx> MirBorrowckCtxt<'_, 'infcx, 'tcx> {
    pub(crate) fn report_opaque_type_errors(&mut self, errors: Vec<DeferredOpaqueTypeError<'tcx>>) {
        if errors.is_empty() {
            return;
        }

        let infcx = self.infcx;
        let mut guar = None;
        let mut last_unexpected_hidden_region: Option<(Span, Ty<'_>, ty::OpaqueTypeKey<'tcx>)> =
            None;
        for error in errors {
            guar = Some(match error {
                DeferredOpaqueTypeError::InvalidOpaqueTypeArgs(err) => err.report(infcx),
                DeferredOpaqueTypeError::LifetimeMismatchOpaqueParam(err) => {
                    infcx.dcx().emit_err(err)
                }
                DeferredOpaqueTypeError::UnexpectedHiddenRegion {
                    opaque_type_key,
                    hidden_type,
                    member_region,
                } => {
                    let named_ty =
                        self.regioncx.name_regions_for_member_constraint(infcx.tcx, hidden_type.ty);
                    let named_key = self
                        .regioncx
                        .name_regions_for_member_constraint(infcx.tcx, opaque_type_key);
                    let named_region =
                        self.regioncx.name_regions_for_member_constraint(infcx.tcx, member_region);
                    let diag = unexpected_hidden_region_diagnostic(
                        infcx,
                        self.mir_def_id(),
                        hidden_type.span,
                        named_ty,
                        named_region,
                        named_key,
                    );
                    if last_unexpected_hidden_region
                        != Some((hidden_type.span, named_ty, named_key))
                    {
                        last_unexpected_hidden_region =
                            Some((hidden_type.span, named_ty, named_key));
                        diag.emit()
                    } else {
                        diag.delay_as_bug()
                    }
                }
                DeferredOpaqueTypeError::NonDefiningUseInDefiningScope {
                    span,
                    opaque_type_key,
                } => infcx.dcx().span_err(
                    span,
                    format!(
                        "non-defining use of `{}` in the defining scope",
                        Ty::new_opaque(
                            infcx.tcx,
                            opaque_type_key.def_id.to_def_id(),
                            opaque_type_key.args
                        )
                    ),
                ),
            });
        }
        let guar = guar.unwrap();
        self.root_cx.set_tainted_by_errors(guar);
        self.infcx.set_tainted_by_errors(guar);
    }

    /// Try to note when an opaque is involved in a borrowck error and that
    /// opaque captures lifetimes due to edition 2024.
    // FIXME: This code is otherwise somewhat general, and could easily be adapted
    // to explain why other things overcapture... like async fn and RPITITs.
    pub(crate) fn note_due_to_edition_2024_opaque_capture_rules(
        &self,
        borrow: &BorrowData<'tcx>,
        diag: &mut Diag<'_>,
    ) {
        // We look at all the locals. Why locals? Because it's the best thing
        // I could think of that's correlated with the *instantiated* higher-ranked
        // binder for calls, since we don't really store those anywhere else.
        for ty in self.body.local_decls.iter().map(|local| local.ty) {
            if !ty.has_opaque_types() {
                continue;
            }

            let tcx = self.infcx.tcx;
            let ControlFlow::Break((opaque_def_id, offending_region_idx, location)) = ty
                .visit_with(&mut FindOpaqueRegion {
                    regioncx: &self.regioncx,
                    tcx,
                    borrow_region: borrow.region,
                })
            else {
                continue;
            };

            // If an opaque explicitly captures a lifetime, then no need to point it out.
            // FIXME: We should be using a better heuristic for `use<>`.
            if tcx.rendered_precise_capturing_args(opaque_def_id).is_some() {
                continue;
            }

            // If one of the opaque's bounds mentions the region, then no need to
            // point it out, since it would've been captured on edition 2021 as well.
            //
            // Also, while we're at it, collect all the lifetimes that the opaque
            // *does* mention. We'll use that for the `+ use<'a>` suggestion below.
            let mut visitor = CheckExplicitRegionMentionAndCollectGenerics {
                tcx,
                generics: tcx.generics_of(opaque_def_id),
                offending_region_idx,
                seen_opaques: [opaque_def_id].into_iter().collect(),
                seen_lifetimes: Default::default(),
            };
            if tcx
                .explicit_item_bounds(opaque_def_id)
                .skip_binder()
                .visit_with(&mut visitor)
                .is_break()
            {
                continue;
            }

            // If we successfully located a terminator, then point it out
            // and provide a suggestion if it's local.
            match self.body.stmt_at(location) {
                Either::Right(mir::Terminator { source_info, .. }) => {
                    diag.span_note(
                        source_info.span,
                        "this call may capture more lifetimes than intended, \
                        because Rust 2024 has adjusted the `impl Trait` lifetime capture rules",
                    );
                    let mut captured_args = visitor.seen_lifetimes;
                    // Add in all of the type and const params, too.
                    // Ordering here is kinda strange b/c we're walking backwards,
                    // but we're trying to provide *a* suggestion, not a nice one.
                    let mut next_generics = Some(visitor.generics);
                    let mut any_synthetic = false;
                    while let Some(generics) = next_generics {
                        for param in &generics.own_params {
                            if param.kind.is_ty_or_const() {
                                captured_args.insert(param.def_id);
                            }
                            if param.kind.is_synthetic() {
                                any_synthetic = true;
                            }
                        }
                        next_generics = generics.parent.map(|def_id| tcx.generics_of(def_id));
                    }

                    if let Some(opaque_def_id) = opaque_def_id.as_local()
                        && let hir::OpaqueTyOrigin::FnReturn { parent, .. } =
                            tcx.hir_expect_opaque_ty(opaque_def_id).origin
                    {
                        if let Some(sugg) = impl_trait_overcapture_suggestion(
                            tcx,
                            opaque_def_id,
                            parent,
                            captured_args,
                        ) {
                            sugg.add_to_diag(diag);
                        }
                    } else {
                        diag.span_help(
                            tcx.def_span(opaque_def_id),
                            format!(
                                "if you can modify this crate, add a precise \
                                capturing bound to avoid overcapturing: `+ use<{}>`",
                                if any_synthetic {
                                    "/* Args */".to_string()
                                } else {
                                    captured_args
                                        .into_iter()
                                        .map(|def_id| tcx.item_name(def_id))
                                        .join(", ")
                                }
                            ),
                        );
                    }
                    return;
                }
                Either::Left(_) => {}
            }
        }
    }
}

/// This visitor contains the bulk of the logic for this lint.
struct FindOpaqueRegion<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,
    borrow_region: ty::RegionVid,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for FindOpaqueRegion<'_, 'tcx> {
    type Result = ControlFlow<(DefId, usize, Location), ()>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        // If we find an opaque in a local ty, then for each of its captured regions,
        // try to find a path between that captured regions and our borrow region...
        if let ty::Alias(ty::Opaque, opaque) = *ty.kind()
            && let hir::OpaqueTyOrigin::FnReturn { parent, in_trait_or_impl: None } =
                self.tcx.opaque_ty_origin(opaque.def_id)
        {
            let variances = self.tcx.variances_of(opaque.def_id);
            for (idx, (arg, variance)) in std::iter::zip(opaque.args, variances).enumerate() {
                // Skip uncaptured args.
                if *variance == ty::Bivariant {
                    continue;
                }
                // We only care about regions.
                let Some(opaque_region) = arg.as_region() else {
                    continue;
                };
                // Don't try to convert a late-bound region, which shouldn't exist anyways (yet).
                if opaque_region.is_bound() {
                    continue;
                }
                let opaque_region_vid = self.regioncx.to_region_vid(opaque_region);

                // Find a path between the borrow region and our opaque capture.
                if let Some(path) = self
                    .regioncx
                    .constraint_path_between_regions(self.borrow_region, opaque_region_vid)
                {
                    for constraint in path {
                        // If we find a call in this path, then check if it defines the opaque.
                        if let ConstraintCategory::CallArgument(Some(call_ty)) = constraint.category
                            && let ty::FnDef(call_def_id, _) = *call_ty.kind()
                            // This function defines the opaque :D
                            && call_def_id == parent
                            && let Locations::Single(location) = constraint.locations
                        {
                            return ControlFlow::Break((opaque.def_id, idx, location));
                        }
                    }
                }
            }
        }

        ty.super_visit_with(self)
    }
}

struct CheckExplicitRegionMentionAndCollectGenerics<'tcx> {
    tcx: TyCtxt<'tcx>,
    generics: &'tcx ty::Generics,
    offending_region_idx: usize,
    seen_opaques: FxIndexSet<DefId>,
    seen_lifetimes: FxIndexSet<DefId>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for CheckExplicitRegionMentionAndCollectGenerics<'tcx> {
    type Result = ControlFlow<(), ()>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        match *ty.kind() {
            ty::Alias(ty::Opaque, opaque) => {
                if self.seen_opaques.insert(opaque.def_id) {
                    for (bound, _) in self
                        .tcx
                        .explicit_item_bounds(opaque.def_id)
                        .iter_instantiated_copied(self.tcx, opaque.args)
                    {
                        bound.visit_with(self)?;
                    }
                }
                ControlFlow::Continue(())
            }
            _ => ty.super_visit_with(self),
        }
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> Self::Result {
        match r.kind() {
            ty::ReEarlyParam(param) => {
                if param.index as usize == self.offending_region_idx {
                    ControlFlow::Break(())
                } else {
                    self.seen_lifetimes.insert(self.generics.region_param(param, self.tcx).def_id);
                    ControlFlow::Continue(())
                }
            }
            _ => ControlFlow::Continue(()),
        }
    }
}
