//! Error reporting machinery for lifetime errors.

use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{Applicability, Diag, ErrorGuaranteed, MultiSpan};
use rustc_hir as hir;
use rustc_hir::GenericBound::Trait;
use rustc_hir::QPath::Resolved;
use rustc_hir::WherePredicateKind::BoundPredicate;
use rustc_hir::def::Res::Def;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::VisitorExt;
use rustc_hir::{PolyTraitRef, TyKind, WhereBoundPredicate};
use rustc_infer::infer::{NllRegionVariableOrigin, RelateParamBound};
use rustc_middle::bug;
use rustc_middle::hir::place::PlaceBase;
use rustc_middle::mir::{AnnotationSource, ConstraintCategory, ReturnConstraint};
use rustc_middle::ty::{
    self, GenericArgs, Region, RegionVid, Ty, TyCtxt, TypeFoldable, TypeVisitor, fold_regions,
};
use rustc_span::{Ident, Span, kw};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::error_reporting::infer::nice_region_error::{
    self, HirTraitObjectVisitor, NiceRegionError, TraitObjectVisitor, find_anon_type,
    find_param_with_region, suggest_adding_lifetime_params,
};
use rustc_trait_selection::error_reporting::infer::region::unexpected_hidden_region_diagnostic;
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::{Obligation, ObligationCtxt};
use tracing::{debug, instrument, trace};

use super::{OutlivesSuggestionBuilder, RegionName, RegionNameSource};
use crate::nll::ConstraintDescription;
use crate::region_infer::values::RegionElement;
use crate::region_infer::{BlameConstraint, TypeTest};
use crate::session_diagnostics::{
    FnMutError, FnMutReturnTypeErr, GenericDoesNotLiveLongEnough, LifetimeOutliveErr,
    LifetimeReturnCategoryErr, RequireStaticErr, VarHereDenote,
};
use crate::universal_regions::DefiningTy;
use crate::{MirBorrowckCtxt, borrowck_errors, fluent_generated as fluent};

impl<'tcx> ConstraintDescription for ConstraintCategory<'tcx> {
    fn description(&self) -> &'static str {
        // Must end with a space. Allows for empty names to be provided.
        match self {
            ConstraintCategory::Assignment => "assignment ",
            ConstraintCategory::Return(_) => "returning this value ",
            ConstraintCategory::Yield => "yielding this value ",
            ConstraintCategory::UseAsConst => "using this value as a constant ",
            ConstraintCategory::UseAsStatic => "using this value as a static ",
            ConstraintCategory::Cast { is_implicit_coercion: false, .. } => "cast ",
            ConstraintCategory::Cast { is_implicit_coercion: true, .. } => "coercion ",
            ConstraintCategory::CallArgument(_) => "argument ",
            ConstraintCategory::TypeAnnotation(AnnotationSource::GenericArg) => "generic argument ",
            ConstraintCategory::TypeAnnotation(_) => "type annotation ",
            ConstraintCategory::SizedBound => "proving this value is `Sized` ",
            ConstraintCategory::CopyBound => "copying this value ",
            ConstraintCategory::OpaqueType => "opaque type ",
            ConstraintCategory::ClosureUpvar(_) => "closure capture ",
            ConstraintCategory::Usage => "this usage ",
            ConstraintCategory::Predicate(_)
            | ConstraintCategory::Boring
            | ConstraintCategory::BoringNoLocation
            | ConstraintCategory::Internal
            | ConstraintCategory::IllegalUniverse => "",
        }
    }
}

/// A collection of errors encountered during region inference. This is needed to efficiently
/// report errors after borrow checking.
///
/// Usually we expect this to either be empty or contain a small number of items, so we can avoid
/// allocation most of the time.
pub(crate) struct RegionErrors<'tcx>(Vec<(RegionErrorKind<'tcx>, ErrorGuaranteed)>, TyCtxt<'tcx>);

impl<'tcx> RegionErrors<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self(vec![], tcx)
    }
    #[track_caller]
    pub(crate) fn push(&mut self, val: impl Into<RegionErrorKind<'tcx>>) {
        let val = val.into();
        let guar = self.1.sess.dcx().delayed_bug(format!("{val:?}"));
        self.0.push((val, guar));
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub(crate) fn into_iter(
        self,
    ) -> impl Iterator<Item = (RegionErrorKind<'tcx>, ErrorGuaranteed)> {
        self.0.into_iter()
    }
    pub(crate) fn has_errors(&self) -> Option<ErrorGuaranteed> {
        self.0.get(0).map(|x| x.1)
    }
}

impl std::fmt::Debug for RegionErrors<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RegionErrors").field(&self.0).finish()
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
pub(crate) struct ErrorConstraintInfo<'tcx> {
    // fr: outlived_fr
    pub(super) fr: RegionVid,
    pub(super) outlived_fr: RegionVid,

    // Category and span for best blame constraint
    pub(super) category: ConstraintCategory<'tcx>,
    pub(super) span: Span,
}

impl<'infcx, 'tcx> MirBorrowckCtxt<'_, 'infcx, 'tcx> {
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

    /// Map the regions in the type to named regions, where possible.
    fn name_regions<T>(&self, tcx: TyCtxt<'tcx>, ty: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        fold_regions(tcx, ty, |region, _| match *region {
            ty::ReVar(vid) => self.to_error_region(vid).unwrap_or(region),
            _ => region,
        })
    }

    /// Returns `true` if a closure is inferred to be an `FnMut` closure.
    fn is_closure_fn_mut(&self, fr: RegionVid) -> bool {
        if let Some(ty::ReLateParam(late_param)) = self.to_error_region(fr).as_deref()
            && let ty::LateParamRegionKind::ClosureEnv = late_param.kind
            && let DefiningTy::Closure(_, args) = self.regioncx.universal_regions().defining_ty
        {
            return args.as_closure().kind() == ty::ClosureKind::FnMut;
        }

        false
    }

    // For generic associated types (GATs) which implied 'static requirement
    // from higher-ranked trait bounds (HRTB). Try to locate span of the trait
    // and the span which bounded to the trait for adding 'static lifetime suggestion
    #[allow(rustc::diagnostic_outside_of_impl)]
    fn suggest_static_lifetime_for_gat_from_hrtb(
        &self,
        diag: &mut Diag<'_>,
        lower_bound: RegionVid,
    ) {
        let mut suggestions = vec![];
        let tcx = self.infcx.tcx;

        // find generic associated types in the given region 'lower_bound'
        let gat_id_and_generics = self
            .regioncx
            .placeholders_contained_in(lower_bound)
            .map(|placeholder| {
                if let Some(id) = placeholder.bound.kind.get_id()
                    && let Some(placeholder_id) = id.as_local()
                    && let gat_hir_id = tcx.local_def_id_to_hir_id(placeholder_id)
                    && let Some(generics_impl) =
                        tcx.parent_hir_node(tcx.parent_hir_id(gat_hir_id)).generics()
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
                let BoundPredicate(WhereBoundPredicate { bound_generic_params, bounds, .. }) =
                    pred.kind
                else {
                    continue;
                };
                if bound_generic_params
                    .iter()
                    .rfind(|bgp| tcx.local_def_id_to_hir_id(bgp.def_id) == *gat_hir_id)
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
            let Trait(PolyTraitRef { trait_ref, span: trait_span, .. }) = bound else {
                return;
            };
            diag.span_note(*trait_span, fluent::borrowck_limitations_implies_static);
            let Some(generics_fn) = tcx.hir_get_generics(self.body.source.def_id().expect_local())
            else {
                return;
            };
            let Def(_, trait_res_defid) = trait_ref.path.res else {
                return;
            };
            debug!(?generics_fn);
            generics_fn.predicates.iter().for_each(|predicate| {
                let BoundPredicate(WhereBoundPredicate { bounded_ty, bounds, .. }) = predicate.kind
                else {
                    return;
                };
                bounds.iter().for_each(|bd| {
                    if let Trait(PolyTraitRef { trait_ref: tr_ref, .. }) = bd
                        && let Def(_, res_defid) = tr_ref.path.res
                        && res_defid == trait_res_defid // trait id matches
                        && let TyKind::Path(Resolved(_, path)) = bounded_ty.kind
                        && let Def(_, defid) = path.res
                        && generics_fn.params
                            .iter()
                            .rfind(|param| param.def_id.to_def_id() == defid)
                            .is_some()
                    {
                        suggestions.push((predicate.span.shrink_to_hi(), " + 'static".to_string()));
                    }
                });
            });
        });
        if suggestions.len() > 0 {
            suggestions.dedup();
            diag.multipart_suggestion_verbose(
                fluent::borrowck_restrict_to_static,
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

        for (nll_error, _) in nll_errors.into_iter() {
            match nll_error {
                RegionErrorKind::TypeTestError { type_test } => {
                    // Try to convert the lower-bound region into something named we can print for
                    // the user.
                    let lower_bound_region = self.to_error_region(type_test.lower_bound);

                    let type_test_span = type_test.span;

                    if let Some(lower_bound_region) = lower_bound_region {
                        let generic_ty = self.name_regions(
                            self.infcx.tcx,
                            type_test.generic_kind.to_ty(self.infcx.tcx),
                        );
                        let origin = RelateParamBound(type_test_span, generic_ty, None);
                        self.buffer_error(self.infcx.err_ctxt().construct_generic_bound_failure(
                            self.body.source.def_id().expect_local(),
                            type_test_span,
                            Some(origin),
                            self.name_regions(self.infcx.tcx, type_test.generic_kind),
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
                        let mut diag = self.dcx().create_err(GenericDoesNotLiveLongEnough {
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
                    let named_ty =
                        self.regioncx.name_regions_for_member_constraint(self.infcx.tcx, hidden_ty);
                    let named_key =
                        self.regioncx.name_regions_for_member_constraint(self.infcx.tcx, key);
                    let named_region = self
                        .regioncx
                        .name_regions_for_member_constraint(self.infcx.tcx, member_region);
                    let diag = unexpected_hidden_region_diagnostic(
                        self.infcx,
                        self.mir_def_id(),
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
    /// Here we would be invoked with `fr = 'a` and `outlived_fr = 'b`.
    // FIXME: make this translatable
    #[allow(rustc::diagnostic_outside_of_impl)]
    #[allow(rustc::untranslatable_diagnostic)]
    pub(crate) fn report_region_error(
        &mut self,
        fr: RegionVid,
        fr_origin: NllRegionVariableOrigin,
        outlived_fr: RegionVid,
        outlives_suggestion: &mut OutlivesSuggestionBuilder,
    ) {
        debug!("report_region_error(fr={:?}, outlived_fr={:?})", fr, outlived_fr);

        let (blame_constraint, path) = self.regioncx.best_blame_constraint(fr, fr_origin, |r| {
            self.regioncx.provides_universal_region(r, fr, outlived_fr)
        });
        let BlameConstraint { category, cause, variance_info, .. } = blame_constraint;

        debug!("report_region_error: category={:?} {:?} {:?}", category, cause, variance_info);

        // Check if we can use one of the "nice region errors".
        if let (Some(f), Some(o)) = (self.to_error_region(fr), self.to_error_region(outlived_fr)) {
            let infer_err = self.infcx.err_ctxt();
            let nice =
                NiceRegionError::new_from_span(&infer_err, self.mir_def_id(), cause.span, o, f);
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

        let errci = ErrorConstraintInfo { fr, outlived_fr, category, span: cause.span };

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
                    ty::RawPtr(ty, mutbl) => {
                        assert_eq!(*mutbl, hir::Mutability::Mut);
                        (
                            format!("a mutable pointer to `{}`", ty),
                            "mutable pointers are invariant over their type parameter".to_string(),
                        )
                    }
                    ty::Ref(_, inner_ty, mutbl) => {
                        assert_eq!(*mutbl, hir::Mutability::Mut);
                        (
                            format!("a mutable reference to `{inner_ty}`"),
                            "mutable references are invariant over their type parameter"
                                .to_string(),
                        )
                    }
                    ty::Adt(adt, args) => {
                        let generic_arg = args[param_index as usize];
                        let identity_args =
                            GenericArgs::identity_for_item(self.infcx.tcx, adt.did());
                        let base_ty = Ty::new_adt(self.infcx.tcx, *adt, identity_args);
                        let base_generic_arg = identity_args[param_index as usize];
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
                        let identity_args = GenericArgs::identity_for_item(self.infcx.tcx, *def_id);
                        let desc = format!("a function pointer to `{name}`");
                        let note = format!(
                            "the function `{name}` is invariant over the parameter `{}`",
                            identity_args[param_index as usize]
                        );
                        (desc, note)
                    }
                    _ => panic!("Unexpected type {ty:?}"),
                };
                diag.note(format!("requirement occurs because of {desc}",));
                diag.note(note);
                diag.help("see <https://doc.rust-lang.org/nomicon/subtyping.html> for more information about variance");
            }
        }

        self.add_placeholder_from_predicate_note(&mut diag, &path);
        self.add_sized_or_copy_bound_info(&mut diag, category, &path);

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
    #[allow(rustc::diagnostic_outside_of_impl)] // FIXME
    fn report_fnmut_error(
        &self,
        errci: &ErrorConstraintInfo<'tcx>,
        kind: ReturnConstraint,
    ) -> Diag<'infcx> {
        let ErrorConstraintInfo { outlived_fr, span, .. } = errci;

        let mut output_ty = self.regioncx.universal_regions().unnormalized_output_ty;
        if let ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }) = *output_ty.kind() {
            output_ty = self.infcx.tcx.type_of(def_id).instantiate_identity()
        };

        debug!("report_fnmut_error: output_ty={:?}", output_ty);

        let err = FnMutError {
            span: *span,
            ty_err: match output_ty.kind() {
                ty::Coroutine(def, ..) if self.infcx.tcx.coroutine_is_async(*def) => {
                    FnMutReturnTypeErr::ReturnAsyncBlock { span: *span }
                }
                _ if output_ty.contains_closure() => {
                    FnMutReturnTypeErr::ReturnClosure { span: *span }
                }
                _ => FnMutReturnTypeErr::ReturnRef { span: *span },
            },
        };

        let mut diag = self.dcx().create_err(err);

        if let ReturnConstraint::ClosureUpvar(upvar_field) = kind {
            let def_id = match self.regioncx.universal_regions().defining_ty {
                DefiningTy::Closure(def_id, _) => def_id,
                ty => bug!("unexpected DefiningTy {:?}", ty),
            };

            let captured_place = &self.upvars[upvar_field.index()].place;
            let defined_hir = match captured_place.base {
                PlaceBase::Local(hirid) => Some(hirid),
                PlaceBase::Upvar(upvar) => Some(upvar.var_path.hir_id),
                _ => None,
            };

            if let Some(def_hir) = defined_hir {
                let upvars_map = self.infcx.tcx.upvars_mentioned(def_id).unwrap();
                let upvar_def_span = self.infcx.tcx.hir_span(def_hir);
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
    fn report_escaping_data_error(&self, errci: &ErrorConstraintInfo<'tcx>) -> Diag<'infcx> {
        let ErrorConstraintInfo { span, category, .. } = errci;

        let fr_name_and_span = self.regioncx.get_var_name_and_span_for_region(
            self.infcx.tcx,
            self.body,
            &self.local_names,
            &self.upvars,
            errci.fr,
        );
        let outlived_fr_name_and_span = self.regioncx.get_var_name_and_span_for_region(
            self.infcx.tcx,
            self.body,
            &self.local_names,
            &self.upvars,
            errci.outlived_fr,
        );

        let escapes_from =
            self.infcx.tcx.def_descr(self.regioncx.universal_regions().defining_ty.def_id());

        // Revert to the normal error in these cases.
        // Assignments aren't "escapes" in function items.
        if (fr_name_and_span.is_none() && outlived_fr_name_and_span.is_none())
            || (*category == ConstraintCategory::Assignment
                && self.regioncx.universal_regions().defining_ty.is_fn_def())
            || self.regioncx.universal_regions().defining_ty.is_const()
        {
            return self.report_general_error(errci);
        }

        let mut diag =
            borrowck_errors::borrowed_data_escapes_closure(self.infcx.tcx, *span, escapes_from);

        if let Some((Some(outlived_fr_name), outlived_fr_span)) = outlived_fr_name_and_span {
            // FIXME: make this translatable
            #[allow(rustc::diagnostic_outside_of_impl)]
            #[allow(rustc::untranslatable_diagnostic)]
            diag.span_label(
                outlived_fr_span,
                format!("`{outlived_fr_name}` declared here, outside of the {escapes_from} body",),
            );
        }

        // FIXME: make this translatable
        #[allow(rustc::diagnostic_outside_of_impl)]
        #[allow(rustc::untranslatable_diagnostic)]
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

                // FIXME: make this translatable
                #[allow(rustc::diagnostic_outside_of_impl)]
                #[allow(rustc::untranslatable_diagnostic)]
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
    #[allow(rustc::diagnostic_outside_of_impl)] // FIXME
    fn report_general_error(&self, errci: &ErrorConstraintInfo<'tcx>) -> Diag<'infcx> {
        let ErrorConstraintInfo { fr, outlived_fr, span, category, .. } = errci;

        let mir_def_name = self.infcx.tcx.def_descr(self.mir_def_id().to_def_id());

        let err = LifetimeOutliveErr { span: *span };
        let mut diag = self.dcx().create_err(err);

        // In certain scenarios, such as the one described in issue #118021,
        // we might encounter a lifetime that cannot be named.
        // These situations are bound to result in errors.
        // To prevent an immediate ICE, we opt to create a dummy name instead.
        let fr_name = self.give_region_a_name(*fr).unwrap_or(RegionName {
            name: kw::UnderscoreLifetime,
            source: RegionNameSource::Static,
        });
        fr_name.highlight_region_name(&mut diag);
        let outlived_fr_name = self.give_region_a_name(*outlived_fr).unwrap();
        outlived_fr_name.highlight_region_name(&mut diag);

        let err_category = if matches!(category, ConstraintCategory::Return(_))
            && self.regioncx.universal_regions().is_local_free_region(*outlived_fr)
        {
            LifetimeReturnCategoryErr::WrongReturn {
                span: *span,
                mir_def_name,
                outlived_fr_name,
                fr_name: &fr_name,
            }
        } else {
            LifetimeReturnCategoryErr::ShortReturn {
                span: *span,
                category_desc: category.description(),
                free_region_name: &fr_name,
                outlived_fr_name,
            }
        };

        diag.subdiagnostic(err_category);

        self.add_static_impl_trait_suggestion(&mut diag, *fr, fr_name, *outlived_fr);
        self.suggest_adding_lifetime_params(&mut diag, *fr, *outlived_fr);
        self.suggest_move_on_borrowing_closure(&mut diag);
        self.suggest_deref_closure_return(&mut diag);

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
    #[allow(rustc::diagnostic_outside_of_impl)]
    #[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
    fn add_static_impl_trait_suggestion(
        &self,
        diag: &mut Diag<'_>,
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
            let suitable_region = self.infcx.tcx.is_suitable_region(self.mir_def_id(), f);
            let Some(suitable_region) = suitable_region else {
                return;
            };

            let fn_returns = self.infcx.tcx.return_type_impl_or_dyn_traits(suitable_region.scope);

            let param = if let Some(param) =
                find_param_with_region(self.infcx.tcx, self.mir_def_id(), f, outlived_f)
            {
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
                    Some(suitable_region.scope),
                );
                return;
            }

            let Some((alias_tys, alias_span, lt_addition_span)) = self
                .infcx
                .tcx
                .return_type_impl_or_dyn_traits_with_type_alias(suitable_region.scope)
            else {
                return;
            };

            // in case the return type of the method is a type alias
            let mut spans_suggs: Vec<_> = Vec::new();
            for alias_ty in alias_tys {
                if alias_ty.span.desugaring_kind().is_some() {
                    // Skip `async` desugaring `impl Future`.
                }
                if let TyKind::TraitObject(_, lt) = alias_ty.kind {
                    if lt.res == hir::LifetimeName::ImplicitObjectLifetimeDefault {
                        spans_suggs.push((lt.ident.span.shrink_to_hi(), " + 'a".to_string()));
                    } else {
                        spans_suggs.push((lt.ident.span, "'a".to_string()));
                    }
                }
            }

            if let Some(lt_addition_span) = lt_addition_span {
                spans_suggs.push((lt_addition_span, "'a, ".to_string()));
            } else {
                spans_suggs.push((alias_span.shrink_to_hi(), "<'a>".to_string()));
            }

            diag.multipart_suggestion_verbose(
                format!(
                    "to declare that the trait object {captures}, you can add a lifetime parameter `'a` in the type alias"
                ),
                spans_suggs,
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn maybe_suggest_constrain_dyn_trait_impl(
        &self,
        diag: &mut Diag<'_>,
        f: Region<'tcx>,
        o: Region<'tcx>,
        category: &ConstraintCategory<'tcx>,
    ) {
        if !o.is_static() {
            return;
        }

        let tcx = self.infcx.tcx;

        let instance = if let ConstraintCategory::CallArgument(Some(func_ty)) = category {
            let (fn_did, args) = match func_ty.kind() {
                ty::FnDef(fn_did, args) => (fn_did, args),
                _ => return,
            };
            debug!(?fn_did, ?args);

            // Only suggest this on function calls, not closures
            let ty = tcx.type_of(fn_did).instantiate_identity();
            debug!("ty: {:?}, ty.kind: {:?}", ty, ty.kind());
            if let ty::Closure(_, _) = ty.kind() {
                return;
            }

            if let Ok(Some(instance)) = ty::Instance::try_resolve(
                tcx,
                self.infcx.typing_env(self.infcx.param_env),
                *fn_did,
                self.infcx.resolve_vars_if_possible(args),
            ) {
                instance
            } else {
                return;
            }
        } else {
            return;
        };

        let param = match find_param_with_region(tcx, self.mir_def_id(), f, o) {
            Some(param) => param,
            None => return,
        };
        debug!(?param);

        let mut visitor = TraitObjectVisitor(FxIndexSet::default());
        visitor.visit_ty(param.param_ty);

        let Some((ident, self_ty)) = NiceRegionError::get_impl_ident_and_self_ty_from_trait(
            tcx,
            instance.def_id(),
            &visitor.0,
        ) else {
            return;
        };

        self.suggest_constrain_dyn_trait_in_impl(diag, &visitor.0, ident, self_ty);
    }

    #[allow(rustc::diagnostic_outside_of_impl)]
    #[instrument(skip(self, err), level = "debug")]
    fn suggest_constrain_dyn_trait_in_impl(
        &self,
        err: &mut Diag<'_>,
        found_dids: &FxIndexSet<DefId>,
        ident: Ident,
        self_ty: &hir::Ty<'_>,
    ) -> bool {
        debug!("err: {:#?}", err);
        let mut suggested = false;
        for found_did in found_dids {
            let mut traits = vec![];
            let mut hir_v = HirTraitObjectVisitor(&mut traits, *found_did);
            hir_v.visit_ty_unambig(self_ty);
            debug!("trait spans found: {:?}", traits);
            for span in &traits {
                let mut multi_span: MultiSpan = vec![*span].into();
                multi_span.push_span_label(*span, fluent::borrowck_implicit_static);
                multi_span.push_span_label(ident.span, fluent::borrowck_implicit_static_introduced);
                err.subdiagnostic(RequireStaticErr::UsedImpl { multi_span });
                err.span_suggestion_verbose(
                    span.shrink_to_hi(),
                    fluent::borrowck_implicit_static_relax,
                    " + '_",
                    Applicability::MaybeIncorrect,
                );
                suggested = true;
            }
        }
        suggested
    }

    fn suggest_adding_lifetime_params(&self, diag: &mut Diag<'_>, sub: RegionVid, sup: RegionVid) {
        let (Some(sub), Some(sup)) = (self.to_error_region(sub), self.to_error_region(sup)) else {
            return;
        };

        let Some((ty_sub, _)) = self
            .infcx
            .tcx
            .is_suitable_region(self.mir_def_id(), sub)
            .and_then(|_| find_anon_type(self.infcx.tcx, self.mir_def_id(), sub))
        else {
            return;
        };

        let Some((ty_sup, _)) = self
            .infcx
            .tcx
            .is_suitable_region(self.mir_def_id(), sup)
            .and_then(|_| find_anon_type(self.infcx.tcx, self.mir_def_id(), sup))
        else {
            return;
        };

        suggest_adding_lifetime_params(
            self.infcx.tcx,
            diag,
            self.mir_def_id(),
            sub,
            ty_sup,
            ty_sub,
        );
    }

    #[allow(rustc::diagnostic_outside_of_impl)]
    /// When encountering a lifetime error caused by the return type of a closure, check the
    /// corresponding trait bound and see if dereferencing the closure return value would satisfy
    /// them. If so, we produce a structured suggestion.
    fn suggest_deref_closure_return(&self, diag: &mut Diag<'_>) {
        let tcx = self.infcx.tcx;

        // Get the closure return value and type.
        let closure_def_id = self.mir_def_id();
        let hir::Node::Expr(
            closure_expr @ hir::Expr {
                kind: hir::ExprKind::Closure(hir::Closure { body, .. }), ..
            },
        ) = tcx.hir_node_by_def_id(closure_def_id)
        else {
            return;
        };
        let ty::Closure(_, args) = *tcx.type_of(closure_def_id).instantiate_identity().kind()
        else {
            return;
        };
        let args = args.as_closure();

        // Make sure that the parent expression is a method call.
        let parent_expr_id = tcx.parent_hir_id(self.mir_hir_id());
        let hir::Node::Expr(
            parent_expr @ hir::Expr {
                kind: hir::ExprKind::MethodCall(_, rcvr, call_args, _), ..
            },
        ) = tcx.hir_node(parent_expr_id)
        else {
            return;
        };
        let typeck_results = tcx.typeck(self.mir_def_id());

        // We don't use `ty.peel_refs()` to get the number of `*`s needed to get the root type.
        let liberated_sig = tcx.liberate_late_bound_regions(closure_def_id.to_def_id(), args.sig());
        let mut peeled_ty = liberated_sig.output();
        let mut count = 0;
        while let ty::Ref(_, ref_ty, _) = *peeled_ty.kind() {
            peeled_ty = ref_ty;
            count += 1;
        }
        if !self.infcx.type_is_copy_modulo_regions(self.infcx.param_env, peeled_ty) {
            return;
        }

        // Build a new closure where the return type is an owned value, instead of a ref.
        let closure_sig_as_fn_ptr_ty = Ty::new_fn_ptr(
            tcx,
            ty::Binder::dummy(tcx.mk_fn_sig(
                liberated_sig.inputs().iter().copied(),
                peeled_ty,
                liberated_sig.c_variadic,
                hir::Safety::Safe,
                rustc_abi::ExternAbi::Rust,
            )),
        );
        let closure_ty = Ty::new_closure(
            tcx,
            closure_def_id.to_def_id(),
            ty::ClosureArgs::new(
                tcx,
                ty::ClosureArgsParts {
                    parent_args: args.parent_args(),
                    closure_kind_ty: args.kind_ty(),
                    tupled_upvars_ty: args.tupled_upvars_ty(),
                    closure_sig_as_fn_ptr_ty,
                },
            )
            .args,
        );

        let Some((closure_arg_pos, _)) =
            call_args.iter().enumerate().find(|(_, arg)| arg.hir_id == closure_expr.hir_id)
        else {
            return;
        };
        // Get the type for the parameter corresponding to the argument the closure with the
        // lifetime error we had.
        let Some(method_def_id) = typeck_results.type_dependent_def_id(parent_expr.hir_id) else {
            return;
        };
        let Some(input_arg) = tcx
            .fn_sig(method_def_id)
            .skip_binder()
            .inputs()
            .skip_binder()
            // Methods have a `self` arg, so `pos` is actually `+ 1` to match the method call arg.
            .get(closure_arg_pos + 1)
        else {
            return;
        };
        // If this isn't a param, then we can't substitute a new closure.
        let ty::Param(closure_param) = input_arg.kind() else { return };

        // Get the arguments for the found method, only specifying that `Self` is the receiver type.
        let Some(possible_rcvr_ty) = typeck_results.node_type_opt(rcvr.hir_id) else { return };
        let args = GenericArgs::for_item(tcx, method_def_id, |param, _| {
            if let ty::GenericParamDefKind::Lifetime = param.kind {
                tcx.lifetimes.re_erased.into()
            } else if param.index == 0 && param.name == kw::SelfUpper {
                possible_rcvr_ty.into()
            } else if param.index == closure_param.index {
                closure_ty.into()
            } else {
                self.infcx.var_for_def(parent_expr.span, param)
            }
        });

        let preds = tcx.predicates_of(method_def_id).instantiate(tcx, args);

        let ocx = ObligationCtxt::new(&self.infcx);
        ocx.register_obligations(preds.iter().map(|(pred, span)| {
            trace!(?pred);
            Obligation::misc(tcx, span, self.mir_def_id(), self.infcx.param_env, pred)
        }));

        if ocx.select_all_or_error().is_empty() && count > 0 {
            diag.span_suggestion_verbose(
                tcx.hir_body(*body).value.peel_blocks().span.shrink_to_lo(),
                fluent::borrowck_dereference_suggestion,
                "*".repeat(count),
                Applicability::MachineApplicable,
            );
        }
    }

    #[allow(rustc::diagnostic_outside_of_impl)]
    fn suggest_move_on_borrowing_closure(&self, diag: &mut Diag<'_>) {
        let body = self.infcx.tcx.hir_body_owned_by(self.mir_def_id());
        let expr = &body.value.peel_blocks();
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
                kind,
                ..
            }) => {
                if !matches!(
                    kind,
                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Async,
                        _
                    ),)
                ) {
                    closure_span = Some(expr.span.shrink_to_lo());
                }
            }
            _ => {}
        }
        if let Some(closure_span) = closure_span {
            diag.span_suggestion_verbose(
                closure_span,
                fluent::borrowck_move_closure_suggestion,
                "move ",
                Applicability::MaybeIncorrect,
            );
        }
    }
}
