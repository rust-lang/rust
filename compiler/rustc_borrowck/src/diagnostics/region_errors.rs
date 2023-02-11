#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
//! Error reporting machinery for lifetime errors.

use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{Applicability, Diagnostic, DiagnosticBuilder, ErrorGuaranteed, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def::Res::Def;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::Visitor;
use rustc_hir::GenericBound::Trait;
use rustc_hir::QPath::Resolved;
use rustc_hir::WherePredicate::BoundPredicate;
use rustc_hir::{PolyTraitRef, TyKind, WhereBoundPredicate};
use rustc_infer::infer::{
    error_reporting::nice_region_error::{
        self, find_anon_type, find_param_with_region, suggest_adding_lifetime_params,
        HirTraitObjectVisitor, NiceRegionError, TraitObjectVisitor,
    },
    error_reporting::unexpected_hidden_region_diagnostic,
    NllRegionVariableOrigin, RelateParamBound,
};
use rustc_middle::hir::place::PlaceBase;
use rustc_middle::mir::{ConstraintCategory, ReturnConstraint};
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::TypeVisitor;
use rustc_middle::ty::{self, RegionVid, Ty};
use rustc_middle::ty::{Region, TyCtxt};
use rustc_span::symbol::{kw, Ident};
use rustc_span::{Span, DUMMY_SP};

use crate::borrowck_errors;
use crate::session_diagnostics::{
    FnMutError, FnMutReturnTypeErr, GenericDoesNotLiveLongEnough, LifetimeOutliveErr,
    LifetimeReturnCategoryErr, RequireStaticErr, VarHereDenote,
};

use super::{OutlivesSuggestionBuilder, RegionName};
use crate::region_infer::{BlameConstraint, ExtraConstraintInfo};
use crate::{
    nll::ConstraintDescription,
    region_infer::{values::RegionElement, TypeTest},
    universal_regions::DefiningTy,
    MirBorrowckCtxt,
};

impl<'tcx> ConstraintDescription for ConstraintCategory<'tcx> {
    fn description(&self) -> &'static str {
        // Must end with a space. Allows for empty names to be provided.
        match self {
            ConstraintCategory::Assignment => "assignment ",
            ConstraintCategory::Return(_) => "returning this value ",
            ConstraintCategory::Yield => "yielding this value ",
            ConstraintCategory::UseAsConst => "using this value as a constant ",
            ConstraintCategory::UseAsStatic => "using this value as a static ",
            ConstraintCategory::Cast => "cast ",
            ConstraintCategory::CallArgument(_) => "argument ",
            ConstraintCategory::TypeAnnotation => "type annotation ",
            ConstraintCategory::ClosureBounds => "closure body ",
            ConstraintCategory::SizedBound => "proving this value is `Sized` ",
            ConstraintCategory::CopyBound => "copying this value ",
            ConstraintCategory::OpaqueType => "opaque type ",
            ConstraintCategory::ClosureUpvar(_) => "closure capture ",
            ConstraintCategory::Usage => "this usage ",
            ConstraintCategory::Predicate(_)
            | ConstraintCategory::Boring
            | ConstraintCategory::BoringNoLocation
            | ConstraintCategory::Internal => "",
        }
    }
}

/// A collection of errors encountered during region inference. This is needed to efficiently
/// report errors after borrow checking.
///
/// Usually we expect this to either be empty or contain a small number of items, so we can avoid
/// allocation most of the time.
pub(crate) struct RegionErrors<'tcx>(Vec<RegionErrorKind<'tcx>>, TyCtxt<'tcx>);

impl<'tcx> RegionErrors<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self(vec![], tcx)
    }
    #[track_caller]
    pub fn push(&mut self, val: impl Into<RegionErrorKind<'tcx>>) {
        let val = val.into();
        self.1.sess.delay_span_bug(DUMMY_SP, format!("{val:?}"));
        self.0.push(val);
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn into_iter(self) -> impl Iterator<Item = RegionErrorKind<'tcx>> {
        self.0.into_iter()
    }
}

#[derive(Clone, Debug)]
pub(crate) enum RegionErrorKind<'tcx> {
    /// A generic bound failure for a type test (`T: 'a`).
    TypeTestError { type_test: TypeTest<'tcx> },

    /// An unexpected hidden region for an opaque type.
    UnexpectedHiddenRegion {
        /// The span for the member constraint.
        span: Span,
        /// The hidden type.
        hidden_ty: Ty<'tcx>,
        /// The opaque type.
        key: ty::OpaqueTypeKey<'tcx>,
        /// The unexpected region.
        member_region: ty::Region<'tcx>,
    },

    /// Higher-ranked subtyping error.
    BoundUniversalRegionError {
        /// The placeholder free region.
        longer_fr: RegionVid,
        /// The region element that erroneously must be outlived by `longer_fr`.
        error_element: RegionElement,
        /// The placeholder region.
        placeholder: ty::PlaceholderRegion,
    },

    /// Any other lifetime error.
    RegionError {
        /// The origin of the region.
        fr_origin: NllRegionVariableOrigin,
        /// The region that should outlive `shorter_fr`.
        longer_fr: RegionVid,
        /// The region that should be shorter, but we can't prove it.
        shorter_fr: RegionVid,
        /// Indicates whether this is a reported error. We currently only report the first error
        /// encountered and leave the rest unreported so as not to overwhelm the user.
        is_reported: bool,
    },
}

/// Information about the various region constraints involved in a borrow checker error.
#[derive(Clone, Debug)]
pub struct ErrorConstraintInfo<'tcx> {
    // fr: outlived_fr
    pub(super) fr: RegionVid,
    pub(super) fr_is_local: bool,
    pub(super) outlived_fr: RegionVid,
    pub(super) outlived_fr_is_local: bool,

    // Category and span for best blame constraint
    pub(super) category: ConstraintCategory<'tcx>,
    pub(super) span: Span,
}

impl<'a, 'tcx> MirBorrowckCtxt<'a, 'tcx> {
    /// Converts a region inference variable into a `ty::Region` that
    /// we can use for error reporting. If `r` is universally bound,
    /// then we use the name that we have on record for it. If `r` is
    /// existentially bound, then we check its inferred value and try
    /// to find a good name from that. Returns `None` if we can't find
    /// one (e.g., this is just some random part of the CFG).
    pub(super) fn to_error_region(&self, r: RegionVid) -> Option<ty::Region<'tcx>> {
        self.to_error_region_vid(r).and_then(|r| self.regioncx.region_definition(r).external_name)
    }

    /// Returns the `RegionVid` corresponding to the region returned by
    /// `to_error_region`.
    pub(super) fn to_error_region_vid(&self, r: RegionVid) -> Option<RegionVid> {
        if self.regioncx.universal_regions().is_universal_region(r) {
            Some(r)
        } else {
            // We just want something nameable, even if it's not
            // actually an upper bound.
            let upper_bound = self.regioncx.approx_universal_upper_bound(r);

            if self.regioncx.upper_bound_in_region_scc(r, upper_bound) {
                self.to_error_region_vid(upper_bound)
            } else {
                None
            }
        }
    }

    /// Returns `true` if a closure is inferred to be an `FnMut` closure.
    fn is_closure_fn_mut(&self, fr: RegionVid) -> bool {
        if let Some(ty::ReFree(free_region)) = self.to_error_region(fr).as_deref()
            && let ty::BoundRegionKind::BrEnv = free_region.bound_region
            && let DefiningTy::Closure(_, substs) = self.regioncx.universal_regions().defining_ty
        {
            return substs.as_closure().kind() == ty::ClosureKind::FnMut;
        }

        false
    }

    // For generic associated types (GATs) which implied 'static requirement
    // from higher-ranked trait bounds (HRTB). Try to locate span of the trait
    // and the span which bounded to the trait for adding 'static lifetime suggestion
    fn suggest_static_lifetime_for_gat_from_hrtb(
        &self,
        diag: &mut DiagnosticBuilder<'_, ErrorGuaranteed>,
        lower_bound: RegionVid,
    ) {
        let mut suggestions = vec![];
        let hir = self.infcx.tcx.hir();

        // find generic associated types in the given region 'lower_bound'
        let gat_id_and_generics = self
            .regioncx
            .placeholders_contained_in(lower_bound)
            .map(|placeholder| {
                if let Some(id) = placeholder.name.get_id()
                    && let Some(placeholder_id) = id.as_local()
                    && let gat_hir_id = hir.local_def_id_to_hir_id(placeholder_id)
                    && let Some(generics_impl) = hir.get_parent(gat_hir_id).generics()
                {
                    Some((gat_hir_id, generics_impl))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        debug!(?gat_id_and_generics);

        // find higher-ranked trait bounds bounded to the generic associated types
        let mut hrtb_bounds = vec![];
        gat_id_and_generics.iter().flatten().for_each(|(gat_hir_id, generics)| {
            for pred in generics.predicates {
                let BoundPredicate(
                        WhereBoundPredicate {
                            bound_generic_params,
                            bounds,
                            ..
                        }) = pred else { continue; };
                if bound_generic_params
                    .iter()
                    .rfind(|bgp| hir.local_def_id_to_hir_id(bgp.def_id) == *gat_hir_id)
                    .is_some()
                {
                    for bound in *bounds {
                        hrtb_bounds.push(bound);
                    }
                }
            }
        });
        debug!(?hrtb_bounds);

        hrtb_bounds.iter().for_each(|bound| {
            let Trait(PolyTraitRef { trait_ref, span: trait_span, .. }, _) = bound else { return; };
            diag.span_note(
                *trait_span,
                format!("due to current limitations in the borrow checker, this implies a `'static` lifetime")
            );
            let Some(generics_fn) = hir.get_generics(self.body.source.def_id().expect_local()) else { return; };
            let Def(_, trait_res_defid) = trait_ref.path.res else { return; };
            debug!(?generics_fn);
            generics_fn.predicates.iter().for_each(|predicate| {
                let BoundPredicate(
                    WhereBoundPredicate {
                        span: bounded_span,
                        bounded_ty,
                        bounds,
                        ..
                    }
                ) = predicate else { return; };
                bounds.iter().for_each(|bd| {
                    if let Trait(PolyTraitRef { trait_ref: tr_ref, .. }, _) = bd
                        && let Def(_, res_defid) = tr_ref.path.res
                        && res_defid == trait_res_defid // trait id matches
                        && let TyKind::Path(Resolved(_, path)) = bounded_ty.kind
                        && let Def(_, defid) = path.res
                        && generics_fn.params
                            .iter()
                            .rfind(|param| param.def_id.to_def_id() == defid)
                            .is_some() {
                            suggestions.push((bounded_span.shrink_to_hi(), format!(" + 'static")));
                        }
                });
            });
        });
        if suggestions.len() > 0 {
            suggestions.dedup();
            diag.multipart_suggestion_verbose(
                format!("consider restricting the type parameter to the `'static` lifetime"),
                suggestions,
                Applicability::MaybeIncorrect,
            );
        }
    }

    /// Produces nice borrowck error diagnostics for all the errors collected in `nll_errors`.
    pub(crate) fn report_region_errors(&mut self, nll_errors: RegionErrors<'tcx>) {
        // Iterate through all the errors, producing a diagnostic for each one. The diagnostics are
        // buffered in the `MirBorrowckCtxt`.

        let mut outlives_suggestion = OutlivesSuggestionBuilder::default();
        let mut last_unexpected_hidden_region: Option<(Span, Ty<'_>, ty::OpaqueTypeKey<'tcx>)> =
            None;

        for nll_error in nll_errors.into_iter() {
            match nll_error {
                RegionErrorKind::TypeTestError { type_test } => {
                    // Try to convert the lower-bound region into something named we can print for the user.
                    let lower_bound_region = self.to_error_region(type_test.lower_bound);

                    let type_test_span = type_test.span;

                    if let Some(lower_bound_region) = lower_bound_region {
                        let generic_ty = type_test.generic_kind.to_ty(self.infcx.tcx);
                        let origin = RelateParamBound(type_test_span, generic_ty, None);
                        self.buffer_error(self.infcx.err_ctxt().construct_generic_bound_failure(
                            self.body.source.def_id().expect_local(),
                            type_test_span,
                            Some(origin),
                            type_test.generic_kind,
                            lower_bound_region,
                        ));
                    } else {
                        // FIXME. We should handle this case better. It
                        // indicates that we have e.g., some region variable
                        // whose value is like `'a+'b` where `'a` and `'b` are
                        // distinct unrelated universal regions that are not
                        // known to outlive one another. It'd be nice to have
                        // some examples where this arises to decide how best
                        // to report it; we could probably handle it by
                        // iterating over the universal regions and reporting
                        // an error that multiple bounds are required.
                        let mut diag =
                            self.infcx.tcx.sess.create_err(GenericDoesNotLiveLongEnough {
                                kind: type_test.generic_kind.to_string(),
                                span: type_test_span,
                            });

                        // Add notes and suggestions for the case of 'static lifetime
                        // implied but not specified when a generic associated types
                        // are from higher-ranked trait bounds
                        self.suggest_static_lifetime_for_gat_from_hrtb(
                            &mut diag,
                            type_test.lower_bound,
                        );

                        self.buffer_error(diag);
                    }
                }

                RegionErrorKind::UnexpectedHiddenRegion { span, hidden_ty, key, member_region } => {
                    let named_ty = self.regioncx.name_regions(self.infcx.tcx, hidden_ty);
                    let named_key = self.regioncx.name_regions(self.infcx.tcx, key);
                    let named_region = self.regioncx.name_regions(self.infcx.tcx, member_region);
                    let mut diag = unexpected_hidden_region_diagnostic(
                        self.infcx.tcx,
                        span,
                        named_ty,
                        named_region,
                        named_key,
                    );
                    if last_unexpected_hidden_region != Some((span, named_ty, named_key)) {
                        self.buffer_error(diag);
                        last_unexpected_hidden_region = Some((span, named_ty, named_key));
                    } else {
                        diag.delay_as_bug();
                    }
                }

                RegionErrorKind::BoundUniversalRegionError {
                    longer_fr,
                    placeholder,
                    error_element,
                } => {
                    let error_vid = self.regioncx.region_from_element(longer_fr, &error_element);

                    // Find the code to blame for the fact that `longer_fr` outlives `error_fr`.
                    let (_, cause) = self.regioncx.find_outlives_blame_span(
                        longer_fr,
                        NllRegionVariableOrigin::Placeholder(placeholder),
                        error_vid,
                    );

                    let universe = placeholder.universe;
                    let universe_info = self.regioncx.universe_info(universe);

                    universe_info.report_error(self, placeholder, error_element, cause);
                }

                RegionErrorKind::RegionError { fr_origin, longer_fr, shorter_fr, is_reported } => {
                    if is_reported {
                        self.report_region_error(
                            longer_fr,
                            fr_origin,
                            shorter_fr,
                            &mut outlives_suggestion,
                        );
                    } else {
                        // We only report the first error, so as not to overwhelm the user. See
                        // `RegRegionErrorKind` docs.
                        //
                        // FIXME: currently we do nothing with these, but perhaps we can do better?
                        // FIXME: try collecting these constraints on the outlives suggestion
                        // builder. Does it make the suggestions any better?
                        debug!(
                            "Unreported region error: can't prove that {:?}: {:?}",
                            longer_fr, shorter_fr
                        );
                    }
                }
            }
        }

        // Emit one outlives suggestions for each MIR def we borrowck
        outlives_suggestion.add_suggestion(self);
    }

    /// Report an error because the universal region `fr` was required to outlive
    /// `outlived_fr` but it is not known to do so. For example:
    ///
    /// ```compile_fail
    /// fn foo<'a, 'b>(x: &'a u32) -> &'b u32 { x }
    /// ```
    ///
    /// Here we would be invoked with `fr = 'a` and `outlived_fr = `'b`.
    pub(crate) fn report_region_error(
        &mut self,
        fr: RegionVid,
        fr_origin: NllRegionVariableOrigin,
        outlived_fr: RegionVid,
        outlives_suggestion: &mut OutlivesSuggestionBuilder,
    ) {
        debug!("report_region_error(fr={:?}, outlived_fr={:?})", fr, outlived_fr);

        let (blame_constraint, extra_info) =
            self.regioncx.best_blame_constraint(fr, fr_origin, |r| {
                self.regioncx.provides_universal_region(r, fr, outlived_fr)
            });
        let BlameConstraint { category, cause, variance_info, .. } = blame_constraint;

        debug!("report_region_error: category={:?} {:?} {:?}", category, cause, variance_info);

        // Check if we can use one of the "nice region errors".
        if let (Some(f), Some(o)) = (self.to_error_region(fr), self.to_error_region(outlived_fr)) {
            let infer_err = self.infcx.err_ctxt();
            let nice = NiceRegionError::new_from_span(&infer_err, cause.span, o, f);
            if let Some(diag) = nice.try_report_from_nll() {
                self.buffer_error(diag);
                return;
            }
        }

        let (fr_is_local, outlived_fr_is_local): (bool, bool) = (
            self.regioncx.universal_regions().is_local_free_region(fr),
            self.regioncx.universal_regions().is_local_free_region(outlived_fr),
        );

        debug!(
            "report_region_error: fr_is_local={:?} outlived_fr_is_local={:?} category={:?}",
            fr_is_local, outlived_fr_is_local, category
        );

        let errci = ErrorConstraintInfo {
            fr,
            outlived_fr,
            fr_is_local,
            outlived_fr_is_local,
            category,
            span: cause.span,
        };

        let mut diag = match (category, fr_is_local, outlived_fr_is_local) {
            (ConstraintCategory::Return(kind), true, false) if self.is_closure_fn_mut(fr) => {
                self.report_fnmut_error(&errci, kind)
            }
            (ConstraintCategory::Assignment, true, false)
            | (ConstraintCategory::CallArgument(_), true, false) => {
                let mut db = self.report_escaping_data_error(&errci);

                outlives_suggestion.intermediate_suggestion(self, &errci, &mut db);
                outlives_suggestion.collect_constraint(fr, outlived_fr);

                db
            }
            _ => {
                let mut db = self.report_general_error(&errci);

                outlives_suggestion.intermediate_suggestion(self, &errci, &mut db);
                outlives_suggestion.collect_constraint(fr, outlived_fr);

                db
            }
        };

        match variance_info {
            ty::VarianceDiagInfo::None => {}
            ty::VarianceDiagInfo::Invariant { ty, param_index } => {
                let (desc, note) = match ty.kind() {
                    ty::RawPtr(ty_mut) => {
                        assert_eq!(ty_mut.mutbl, rustc_hir::Mutability::Mut);
                        (
                            format!("a mutable pointer to `{}`", ty_mut.ty),
                            "mutable pointers are invariant over their type parameter".to_string(),
                        )
                    }
                    ty::Ref(_, inner_ty, mutbl) => {
                        assert_eq!(*mutbl, rustc_hir::Mutability::Mut);
                        (
                            format!("a mutable reference to `{inner_ty}`"),
                            "mutable references are invariant over their type parameter"
                                .to_string(),
                        )
                    }
                    ty::Adt(adt, substs) => {
                        let generic_arg = substs[param_index as usize];
                        let identity_substs =
                            InternalSubsts::identity_for_item(self.infcx.tcx, adt.did());
                        let base_ty = self.infcx.tcx.mk_adt(*adt, identity_substs);
                        let base_generic_arg = identity_substs[param_index as usize];
                        let adt_desc = adt.descr();

                        let desc = format!(
                            "the type `{ty}`, which makes the generic argument `{generic_arg}` invariant"
                        );
                        let note = format!(
                            "the {adt_desc} `{base_ty}` is invariant over the parameter `{base_generic_arg}`"
                        );
                        (desc, note)
                    }
                    ty::FnDef(def_id, _) => {
                        let name = self.infcx.tcx.item_name(*def_id);
                        let identity_substs =
                            InternalSubsts::identity_for_item(self.infcx.tcx, *def_id);
                        let desc = format!("a function pointer to `{name}`");
                        let note = format!(
                            "the function `{name}` is invariant over the parameter `{}`",
                            identity_substs[param_index as usize]
                        );
                        (desc, note)
                    }
                    _ => panic!("Unexpected type {ty:?}"),
                };
                diag.note(&format!("requirement occurs because of {desc}",));
                diag.note(&note);
                diag.help("see <https://doc.rust-lang.org/nomicon/subtyping.html> for more information about variance");
            }
        }

        for extra in extra_info {
            match extra {
                ExtraConstraintInfo::PlaceholderFromPredicate(span) => {
                    diag.span_note(span, "due to current limitations in the borrow checker, this implies a `'static` lifetime");
                }
            }
        }

        self.buffer_error(diag);
    }

    /// Report a specialized error when `FnMut` closures return a reference to a captured variable.
    /// This function expects `fr` to be local and `outlived_fr` to not be local.
    ///
    /// ```text
    /// error: captured variable cannot escape `FnMut` closure body
    ///   --> $DIR/issue-53040.rs:15:8
    ///    |
    /// LL |     || &mut v;
    ///    |     -- ^^^^^^ creates a reference to a captured variable which escapes the closure body
    ///    |     |
    ///    |     inferred to be a `FnMut` closure
    ///    |
    ///    = note: `FnMut` closures only have access to their captured variables while they are
    ///            executing...
    ///    = note: ...therefore, returned references to captured variables will escape the closure
    /// ```
    fn report_fnmut_error(
        &self,
        errci: &ErrorConstraintInfo<'tcx>,
        kind: ReturnConstraint,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let ErrorConstraintInfo { outlived_fr, span, .. } = errci;

        let mut output_ty = self.regioncx.universal_regions().unnormalized_output_ty;
        if let ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }) = *output_ty.kind() {
            output_ty = self.infcx.tcx.type_of(def_id)
        };

        debug!("report_fnmut_error: output_ty={:?}", output_ty);

        let err = FnMutError {
            span: *span,
            ty_err: match output_ty.kind() {
                ty::Generator(def, ..) if self.infcx.tcx.generator_is_async(*def) => {
                    FnMutReturnTypeErr::ReturnAsyncBlock { span: *span }
                }
                _ if output_ty.contains_closure() => {
                    FnMutReturnTypeErr::ReturnClosure { span: *span }
                }
                _ => FnMutReturnTypeErr::ReturnRef { span: *span },
            },
        };

        let mut diag = self.infcx.tcx.sess.create_err(err);

        if let ReturnConstraint::ClosureUpvar(upvar_field) = kind {
            let def_id = match self.regioncx.universal_regions().defining_ty {
                DefiningTy::Closure(def_id, _) => def_id,
                ty => bug!("unexpected DefiningTy {:?}", ty),
            };

            let captured_place = &self.upvars[upvar_field.index()].place;
            let defined_hir = match captured_place.place.base {
                PlaceBase::Local(hirid) => Some(hirid),
                PlaceBase::Upvar(upvar) => Some(upvar.var_path.hir_id),
                _ => None,
            };

            if let Some(def_hir) = defined_hir {
                let upvars_map = self.infcx.tcx.upvars_mentioned(def_id).unwrap();
                let upvar_def_span = self.infcx.tcx.hir().span(def_hir);
                let upvar_span = upvars_map.get(&def_hir).unwrap().span;
                diag.subdiagnostic(VarHereDenote::Defined { span: upvar_def_span });
                diag.subdiagnostic(VarHereDenote::Captured { span: upvar_span });
            }
        }

        if let Some(fr_span) = self.give_region_a_name(*outlived_fr).unwrap().span() {
            diag.subdiagnostic(VarHereDenote::FnMutInferred { span: fr_span });
        }

        self.suggest_move_on_borrowing_closure(&mut diag);

        diag
    }

    /// Reports an error specifically for when data is escaping a closure.
    ///
    /// ```text
    /// error: borrowed data escapes outside of function
    ///   --> $DIR/lifetime-bound-will-change-warning.rs:44:5
    ///    |
    /// LL | fn test2<'a>(x: &'a Box<Fn()+'a>) {
    ///    |              - `x` is a reference that is only valid in the function body
    /// LL |     // but ref_obj will not, so warn.
    /// LL |     ref_obj(x)
    ///    |     ^^^^^^^^^^ `x` escapes the function body here
    /// ```
    #[instrument(level = "debug", skip(self))]
    fn report_escaping_data_error(
        &self,
        errci: &ErrorConstraintInfo<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let ErrorConstraintInfo { span, category, .. } = errci;

        let fr_name_and_span = self.regioncx.get_var_name_and_span_for_region(
            self.infcx.tcx,
            &self.body,
            &self.local_names,
            &self.upvars,
            errci.fr,
        );
        let outlived_fr_name_and_span = self.regioncx.get_var_name_and_span_for_region(
            self.infcx.tcx,
            &self.body,
            &self.local_names,
            &self.upvars,
            errci.outlived_fr,
        );

        let (_, escapes_from) = self
            .infcx
            .tcx
            .article_and_description(self.regioncx.universal_regions().defining_ty.def_id());

        // Revert to the normal error in these cases.
        // Assignments aren't "escapes" in function items.
        if (fr_name_and_span.is_none() && outlived_fr_name_and_span.is_none())
            || (*category == ConstraintCategory::Assignment
                && self.regioncx.universal_regions().defining_ty.is_fn_def())
            || self.regioncx.universal_regions().defining_ty.is_const()
        {
            return self.report_general_error(&ErrorConstraintInfo {
                fr_is_local: true,
                outlived_fr_is_local: false,
                ..*errci
            });
        }

        let mut diag =
            borrowck_errors::borrowed_data_escapes_closure(self.infcx.tcx, *span, escapes_from);

        if let Some((Some(outlived_fr_name), outlived_fr_span)) = outlived_fr_name_and_span {
            diag.span_label(
                outlived_fr_span,
                format!("`{outlived_fr_name}` declared here, outside of the {escapes_from} body",),
            );
        }

        if let Some((Some(fr_name), fr_span)) = fr_name_and_span {
            diag.span_label(
                fr_span,
                format!(
                    "`{fr_name}` is a reference that is only valid in the {escapes_from} body",
                ),
            );

            diag.span_label(*span, format!("`{fr_name}` escapes the {escapes_from} body here"));
        }

        // Only show an extra note if we can find an 'error region' for both of the region
        // variables. This avoids showing a noisy note that just mentions 'synthetic' regions
        // that don't help the user understand the error.
        match (self.to_error_region(errci.fr), self.to_error_region(errci.outlived_fr)) {
            (Some(f), Some(o)) => {
                self.maybe_suggest_constrain_dyn_trait_impl(&mut diag, f, o, category);

                let fr_region_name = self.give_region_a_name(errci.fr).unwrap();
                fr_region_name.highlight_region_name(&mut diag);
                let outlived_fr_region_name = self.give_region_a_name(errci.outlived_fr).unwrap();
                outlived_fr_region_name.highlight_region_name(&mut diag);

                diag.span_label(
                    *span,
                    format!(
                        "{}requires that `{}` must outlive `{}`",
                        category.description(),
                        fr_region_name,
                        outlived_fr_region_name,
                    ),
                );
            }
            _ => {}
        }

        diag
    }

    /// Reports a region inference error for the general case with named/synthesized lifetimes to
    /// explain what is happening.
    ///
    /// ```text
    /// error: unsatisfied lifetime constraints
    ///   --> $DIR/regions-creating-enums3.rs:17:5
    ///    |
    /// LL | fn mk_add_bad1<'a,'b>(x: &'a ast<'a>, y: &'b ast<'b>) -> ast<'a> {
    ///    |                -- -- lifetime `'b` defined here
    ///    |                |
    ///    |                lifetime `'a` defined here
    /// LL |     ast::add(x, y)
    ///    |     ^^^^^^^^^^^^^^ function was supposed to return data with lifetime `'a` but it
    ///    |                    is returning data with lifetime `'b`
    /// ```
    fn report_general_error(
        &self,
        errci: &ErrorConstraintInfo<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let ErrorConstraintInfo {
            fr,
            fr_is_local,
            outlived_fr,
            outlived_fr_is_local,
            span,
            category,
            ..
        } = errci;

        let (_, mir_def_name) =
            self.infcx.tcx.article_and_description(self.mir_def_id().to_def_id());

        let err = LifetimeOutliveErr { span: *span };
        let mut diag = self.infcx.tcx.sess.create_err(err);

        let fr_name = self.give_region_a_name(*fr).unwrap();
        fr_name.highlight_region_name(&mut diag);
        let outlived_fr_name = self.give_region_a_name(*outlived_fr).unwrap();
        outlived_fr_name.highlight_region_name(&mut diag);

        let err_category = match (category, outlived_fr_is_local, fr_is_local) {
            (ConstraintCategory::Return(_), true, _) => LifetimeReturnCategoryErr::WrongReturn {
                span: *span,
                mir_def_name,
                outlived_fr_name,
                fr_name: &fr_name,
            },
            _ => LifetimeReturnCategoryErr::ShortReturn {
                span: *span,
                category_desc: category.description(),
                free_region_name: &fr_name,
                outlived_fr_name,
            },
        };

        diag.subdiagnostic(err_category);

        self.add_static_impl_trait_suggestion(&mut diag, *fr, fr_name, *outlived_fr);
        self.suggest_adding_lifetime_params(&mut diag, *fr, *outlived_fr);
        self.suggest_move_on_borrowing_closure(&mut diag);

        diag
    }

    /// Adds a suggestion to errors where an `impl Trait` is returned.
    ///
    /// ```text
    /// help: to allow this `impl Trait` to capture borrowed data with lifetime `'1`, add `'_` as
    ///       a constraint
    ///    |
    /// LL |     fn iter_values_anon(&self) -> impl Iterator<Item=u32> + 'a {
    ///    |                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    /// ```
    fn add_static_impl_trait_suggestion(
        &self,
        diag: &mut Diagnostic,
        fr: RegionVid,
        // We need to pass `fr_name` - computing it again will label it twice.
        fr_name: RegionName,
        outlived_fr: RegionVid,
    ) {
        if let (Some(f), Some(outlived_f)) =
            (self.to_error_region(fr), self.to_error_region(outlived_fr))
        {
            if *outlived_f != ty::ReStatic {
                return;
            }
            let suitable_region = self.infcx.tcx.is_suitable_region(f);
            let Some(suitable_region) = suitable_region else { return; };

            let fn_returns = self.infcx.tcx.return_type_impl_or_dyn_traits(suitable_region.def_id);

            let param = if let Some(param) = find_param_with_region(self.infcx.tcx, f, outlived_f) {
                param
            } else {
                return;
            };

            let lifetime = if f.has_name() { fr_name.name } else { kw::UnderscoreLifetime };

            let arg = match param.param.pat.simple_ident() {
                Some(simple_ident) => format!("argument `{simple_ident}`"),
                None => "the argument".to_string(),
            };
            let captures = format!("captures data from {arg}");

            if !fn_returns.is_empty() {
                nice_region_error::suggest_new_region_bound(
                    self.infcx.tcx,
                    diag,
                    fn_returns,
                    lifetime.to_string(),
                    Some(arg),
                    captures,
                    Some((param.param_ty_span, param.param_ty.to_string())),
                    Some(suitable_region.def_id),
                );
                return;
            }

            let Some((alias_tys, alias_span)) = self
                .infcx
                .tcx
                .return_type_impl_or_dyn_traits_with_type_alias(suitable_region.def_id) else { return; };

            // in case the return type of the method is a type alias
            let mut spans_suggs: Vec<_> = Vec::new();
            for alias_ty in alias_tys {
                if alias_ty.span.desugaring_kind().is_some() {
                    // Skip `async` desugaring `impl Future`.
                    ()
                }
                if let TyKind::TraitObject(_, lt, _) = alias_ty.kind {
                    spans_suggs.push((lt.ident.span.shrink_to_hi(), " + 'a".to_string()));
                }
            }
            spans_suggs.push((alias_span.shrink_to_hi(), "<'a>".to_string()));
            diag.multipart_suggestion_verbose(
                &format!(
                    "to declare that the trait object {captures}, you can add a lifetime parameter `'a` in the type alias"
                ),
                spans_suggs,
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn maybe_suggest_constrain_dyn_trait_impl(
        &self,
        diag: &mut Diagnostic,
        f: Region<'tcx>,
        o: Region<'tcx>,
        category: &ConstraintCategory<'tcx>,
    ) {
        if !o.is_static() {
            return;
        }

        let tcx = self.infcx.tcx;

        let instance = if let ConstraintCategory::CallArgument(Some(func_ty)) = category {
            let (fn_did, substs) = match func_ty.kind() {
                ty::FnDef(fn_did, substs) => (fn_did, substs),
                _ => return,
            };
            debug!(?fn_did, ?substs);

            // Only suggest this on function calls, not closures
            let ty = tcx.type_of(fn_did);
            debug!("ty: {:?}, ty.kind: {:?}", ty, ty.kind());
            if let ty::Closure(_, _) = ty.kind() {
                return;
            }

            if let Ok(Some(instance)) = ty::Instance::resolve(
                tcx,
                self.param_env,
                *fn_did,
                self.infcx.resolve_vars_if_possible(substs),
            ) {
                instance
            } else {
                return;
            }
        } else {
            return;
        };

        let param = match find_param_with_region(tcx, f, o) {
            Some(param) => param,
            None => return,
        };
        debug!(?param);

        let mut visitor = TraitObjectVisitor(FxIndexSet::default());
        visitor.visit_ty(param.param_ty);

        let Some((ident, self_ty)) =
            NiceRegionError::get_impl_ident_and_self_ty_from_trait(tcx, instance.def_id(), &visitor.0) else { return; };

        self.suggest_constrain_dyn_trait_in_impl(diag, &visitor.0, ident, self_ty);
    }

    #[instrument(skip(self, err), level = "debug")]
    fn suggest_constrain_dyn_trait_in_impl(
        &self,
        err: &mut Diagnostic,
        found_dids: &FxIndexSet<DefId>,
        ident: Ident,
        self_ty: &hir::Ty<'_>,
    ) -> bool {
        debug!("err: {:#?}", err);
        let mut suggested = false;
        for found_did in found_dids {
            let mut traits = vec![];
            let mut hir_v = HirTraitObjectVisitor(&mut traits, *found_did);
            hir_v.visit_ty(&self_ty);
            debug!("trait spans found: {:?}", traits);
            for span in &traits {
                let mut multi_span: MultiSpan = vec![*span].into();
                multi_span
                    .push_span_label(*span, "this has an implicit `'static` lifetime requirement");
                multi_span.push_span_label(
                    ident.span,
                    "calling this method introduces the `impl`'s 'static` requirement",
                );
                err.subdiagnostic(RequireStaticErr::UsedImpl { multi_span });
                err.span_suggestion_verbose(
                    span.shrink_to_hi(),
                    "consider relaxing the implicit `'static` requirement",
                    " + '_",
                    Applicability::MaybeIncorrect,
                );
                suggested = true;
            }
        }
        suggested
    }

    fn suggest_adding_lifetime_params(
        &self,
        diag: &mut Diagnostic,
        sub: RegionVid,
        sup: RegionVid,
    ) {
        let (Some(sub), Some(sup)) = (self.to_error_region(sub), self.to_error_region(sup)) else {
            return
        };

        let Some((ty_sub, _)) = self
            .infcx
            .tcx
            .is_suitable_region(sub)
            .and_then(|anon_reg| find_anon_type(self.infcx.tcx, sub, &anon_reg.boundregion)) else {
            return
        };

        let Some((ty_sup, _)) = self
            .infcx
            .tcx
            .is_suitable_region(sup)
            .and_then(|anon_reg| find_anon_type(self.infcx.tcx, sup, &anon_reg.boundregion)) else {
            return
        };

        suggest_adding_lifetime_params(self.infcx.tcx, sub, ty_sup, ty_sub, diag);
    }

    fn suggest_move_on_borrowing_closure(&self, diag: &mut Diagnostic) {
        let map = self.infcx.tcx.hir();
        let body_id = map.body_owned_by(self.mir_def_id());
        let expr = &map.body(body_id).value.peel_blocks();
        let mut closure_span = None::<rustc_span::Span>;
        match expr.kind {
            hir::ExprKind::MethodCall(.., args, _) => {
                for arg in args {
                    if let hir::ExprKind::Closure(hir::Closure {
                        capture_clause: hir::CaptureBy::Ref,
                        ..
                    }) = arg.kind
                    {
                        closure_span = Some(arg.span.shrink_to_lo());
                        break;
                    }
                }
            }
            hir::ExprKind::Closure(hir::Closure {
                capture_clause: hir::CaptureBy::Ref,
                body,
                ..
            }) => {
                let body = map.body(*body);
                if !matches!(body.generator_kind, Some(hir::GeneratorKind::Async(..))) {
                    closure_span = Some(expr.span.shrink_to_lo());
                }
            }
            _ => {}
        }
        if let Some(closure_span) = closure_span {
            diag.span_suggestion_verbose(
                closure_span,
                "consider adding 'move' keyword before the nested closure",
                "move ",
                Applicability::MaybeIncorrect,
            );
        }
    }
}
