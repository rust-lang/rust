//! Error reporting machinery for lifetime errors.

use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_infer::infer::{
    error_reporting::nice_region_error::NiceRegionError,
    error_reporting::unexpected_hidden_region_diagnostic, NllRegionVariableOrigin,
};
use rustc_middle::mir::{ConstraintCategory, ReturnConstraint};
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{self, RegionVid, Ty};
use rustc_span::symbol::{kw, sym};
use rustc_span::{BytePos, Span};

use crate::borrowck_errors;

use super::{OutlivesSuggestionBuilder, RegionName};
use crate::region_infer::BlameConstraint;
use crate::{
    nll::ConstraintDescription,
    region_infer::{values::RegionElement, TypeTest},
    universal_regions::DefiningTy,
    MirBorrowckCtxt,
};

impl ConstraintDescription for ConstraintCategory {
    fn description(&self) -> &'static str {
        // Must end with a space. Allows for empty names to be provided.
        match self {
            ConstraintCategory::Assignment => "assignment ",
            ConstraintCategory::Return(_) => "returning this value ",
            ConstraintCategory::Yield => "yielding this value ",
            ConstraintCategory::UseAsConst => "using this value as a constant ",
            ConstraintCategory::UseAsStatic => "using this value as a static ",
            ConstraintCategory::Cast => "cast ",
            ConstraintCategory::CallArgument => "argument ",
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
crate type RegionErrors<'tcx> = Vec<RegionErrorKind<'tcx>>;

#[derive(Clone, Debug)]
crate enum RegionErrorKind<'tcx> {
    /// A generic bound failure for a type test (`T: 'a`).
    TypeTestError { type_test: TypeTest<'tcx> },

    /// An unexpected hidden region for an opaque type.
    UnexpectedHiddenRegion {
        /// The span for the member constraint.
        span: Span,
        /// The hidden type.
        hidden_ty: Ty<'tcx>,
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
pub struct ErrorConstraintInfo {
    // fr: outlived_fr
    pub(super) fr: RegionVid,
    pub(super) fr_is_local: bool,
    pub(super) outlived_fr: RegionVid,
    pub(super) outlived_fr_is_local: bool,

    // Category and span for best blame constraint
    pub(super) category: ConstraintCategory,
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
        if let Some(ty::ReFree(free_region)) = self.to_error_region(fr) {
            if let ty::BoundRegionKind::BrEnv = free_region.bound_region {
                if let DefiningTy::Closure(_, substs) =
                    self.regioncx.universal_regions().defining_ty
                {
                    return substs.as_closure().kind() == ty::ClosureKind::FnMut;
                }
            }
        }

        false
    }

    /// Produces nice borrowck error diagnostics for all the errors collected in `nll_errors`.
    pub(crate) fn report_region_errors(&mut self, nll_errors: RegionErrors<'tcx>) {
        // Iterate through all the errors, producing a diagnostic for each one. The diagnostics are
        // buffered in the `MirBorrowckCtxt`.

        let mut outlives_suggestion = OutlivesSuggestionBuilder::default();

        for nll_error in nll_errors.into_iter() {
            match nll_error {
                RegionErrorKind::TypeTestError { type_test } => {
                    // Try to convert the lower-bound region into something named we can print for the user.
                    let lower_bound_region = self.to_error_region(type_test.lower_bound);

                    let type_test_span = type_test.locations.span(&self.body);

                    if let Some(lower_bound_region) = lower_bound_region {
                        self.infcx
                            .construct_generic_bound_failure(
                                type_test_span,
                                None,
                                type_test.generic_kind,
                                lower_bound_region,
                            )
                            .buffer(&mut self.errors_buffer);
                    } else {
                        // FIXME. We should handle this case better. It
                        // indicates that we have e.g., some region variable
                        // whose value is like `'a+'b` where `'a` and `'b` are
                        // distinct unrelated univesal regions that are not
                        // known to outlive one another. It'd be nice to have
                        // some examples where this arises to decide how best
                        // to report it; we could probably handle it by
                        // iterating over the universal regions and reporting
                        // an error that multiple bounds are required.
                        self.infcx
                            .tcx
                            .sess
                            .struct_span_err(
                                type_test_span,
                                &format!("`{}` does not live long enough", type_test.generic_kind),
                            )
                            .buffer(&mut self.errors_buffer);
                    }
                }

                RegionErrorKind::UnexpectedHiddenRegion { span, hidden_ty, member_region } => {
                    let named_ty = self.regioncx.name_regions(self.infcx.tcx, hidden_ty);
                    let named_region = self.regioncx.name_regions(self.infcx.tcx, member_region);
                    unexpected_hidden_region_diagnostic(
                        self.infcx.tcx,
                        span,
                        named_ty,
                        named_region,
                    )
                    .buffer(&mut self.errors_buffer);
                }

                RegionErrorKind::BoundUniversalRegionError {
                    longer_fr,
                    placeholder,
                    error_element,
                } => {
                    let error_vid = self.regioncx.region_from_element(longer_fr, &error_element);

                    // Find the code to blame for the fact that `longer_fr` outlives `error_fr`.
                    let (_, cause) = self.regioncx.find_outlives_blame_span(
                        &self.body,
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
    /// ```
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

        let BlameConstraint { category, cause, variance_info, from_closure: _ } =
            self.regioncx.best_blame_constraint(&self.body, fr, fr_origin, |r| {
                self.regioncx.provides_universal_region(r, fr, outlived_fr)
            });

        debug!("report_region_error: category={:?} {:?} {:?}", category, cause, variance_info);
        // Check if we can use one of the "nice region errors".
        if let (Some(f), Some(o)) = (self.to_error_region(fr), self.to_error_region(outlived_fr)) {
            let nice = NiceRegionError::new_from_span(self.infcx, cause.span, o, f);
            if let Some(diag) = nice.try_report_from_nll() {
                diag.buffer(&mut self.errors_buffer);
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
            | (ConstraintCategory::CallArgument, true, false) => {
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
            ty::VarianceDiagInfo::Mut { kind, ty } => {
                let kind_name = match kind {
                    ty::VarianceDiagMutKind::Ref => "reference",
                    ty::VarianceDiagMutKind::RawPtr => "pointer",
                };
                diag.note(&format!("requirement occurs because of a mutable {kind_name} to {ty}",));
                diag.note(&format!("mutable {kind_name}s are invariant over their type parameter"));
                diag.help("see <https://doc.rust-lang.org/nomicon/subtyping.html> for more information about variance");
            }
        }

        diag.buffer(&mut self.errors_buffer);
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
        errci: &ErrorConstraintInfo,
        kind: ReturnConstraint,
    ) -> DiagnosticBuilder<'tcx> {
        let ErrorConstraintInfo { outlived_fr, span, .. } = errci;

        let mut diag = self
            .infcx
            .tcx
            .sess
            .struct_span_err(*span, "captured variable cannot escape `FnMut` closure body");

        let mut output_ty = self.regioncx.universal_regions().unnormalized_output_ty;
        if let ty::Opaque(def_id, _) = *output_ty.kind() {
            output_ty = self.infcx.tcx.type_of(def_id)
        };

        debug!("report_fnmut_error: output_ty={:?}", output_ty);

        let message = match output_ty.kind() {
            ty::Closure(_, _) => {
                "returns a closure that contains a reference to a captured variable, which then \
                 escapes the closure body"
            }
            ty::Adt(def, _) if self.infcx.tcx.is_diagnostic_item(sym::gen_future, def.did) => {
                "returns an `async` block that contains a reference to a captured variable, which then \
                 escapes the closure body"
            }
            _ => "returns a reference to a captured variable which escapes the closure body",
        };

        diag.span_label(*span, message);

        // FIXME(project-rfc-2229#48): This should store a captured_place not a hir id
        if let ReturnConstraint::ClosureUpvar(upvar) = kind {
            let def_id = match self.regioncx.universal_regions().defining_ty {
                DefiningTy::Closure(def_id, _) => def_id,
                ty => bug!("unexpected DefiningTy {:?}", ty),
            };

            let upvar_def_span = self.infcx.tcx.hir().span(upvar);
            let upvar_span = self.infcx.tcx.upvars_mentioned(def_id).unwrap()[&upvar].span;
            diag.span_label(upvar_def_span, "variable defined here");
            diag.span_label(upvar_span, "variable captured here");
        }

        if let Some(fr_span) = self.give_region_a_name(*outlived_fr).unwrap().span() {
            diag.span_label(fr_span, "inferred to be a `FnMut` closure");
        }

        diag.note(
            "`FnMut` closures only have access to their captured variables while they are \
             executing...",
        );
        diag.note("...therefore, they cannot allow references to captured variables to escape");

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
    fn report_escaping_data_error(&self, errci: &ErrorConstraintInfo) -> DiagnosticBuilder<'tcx> {
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
                format!(
                    "`{}` declared here, outside of the {} body",
                    outlived_fr_name, escapes_from
                ),
            );
        }

        if let Some((Some(fr_name), fr_span)) = fr_name_and_span {
            diag.span_label(
                fr_span,
                format!(
                    "`{}` is a reference that is only valid in the {} body",
                    fr_name, escapes_from
                ),
            );

            diag.span_label(*span, format!("`{}` escapes the {} body here", fr_name, escapes_from));
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
    fn report_general_error(&self, errci: &ErrorConstraintInfo) -> DiagnosticBuilder<'tcx> {
        let ErrorConstraintInfo {
            fr,
            fr_is_local,
            outlived_fr,
            outlived_fr_is_local,
            span,
            category,
            ..
        } = errci;

        let mut diag =
            self.infcx.tcx.sess.struct_span_err(*span, "lifetime may not live long enough");

        let (_, mir_def_name) =
            self.infcx.tcx.article_and_description(self.mir_def_id().to_def_id());

        let fr_name = self.give_region_a_name(*fr).unwrap();
        fr_name.highlight_region_name(&mut diag);
        let outlived_fr_name = self.give_region_a_name(*outlived_fr).unwrap();
        outlived_fr_name.highlight_region_name(&mut diag);

        match (category, outlived_fr_is_local, fr_is_local) {
            (ConstraintCategory::Return(_), true, _) => {
                diag.span_label(
                    *span,
                    format!(
                        "{} was supposed to return data with lifetime `{}` but it is returning \
                         data with lifetime `{}`",
                        mir_def_name, outlived_fr_name, fr_name
                    ),
                );
            }
            _ => {
                diag.span_label(
                    *span,
                    format!(
                        "{}requires that `{}` must outlive `{}`",
                        category.description(),
                        fr_name,
                        outlived_fr_name,
                    ),
                );
            }
        }

        self.add_static_impl_trait_suggestion(&mut diag, *fr, fr_name, *outlived_fr);

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
        diag: &mut DiagnosticBuilder<'tcx>,
        fr: RegionVid,
        // We need to pass `fr_name` - computing it again will label it twice.
        fr_name: RegionName,
        outlived_fr: RegionVid,
    ) {
        if let (Some(f), Some(ty::RegionKind::ReStatic)) =
            (self.to_error_region(fr), self.to_error_region(outlived_fr))
        {
            if let Some(&ty::Opaque(did, substs)) = self
                .infcx
                .tcx
                .is_suitable_region(f)
                .map(|r| r.def_id)
                .and_then(|id| self.infcx.tcx.return_type_impl_trait(id))
                .map(|(ty, _)| ty.kind())
            {
                // Check whether or not the impl trait return type is intended to capture
                // data with the static lifetime.
                //
                // eg. check for `impl Trait + 'static` instead of `impl Trait`.
                let has_static_predicate = {
                    let bounds = self.infcx.tcx.explicit_item_bounds(did);

                    let mut found = false;
                    for (bound, _) in bounds {
                        if let ty::PredicateKind::TypeOutlives(ty::OutlivesPredicate(_, r)) =
                            bound.kind().skip_binder()
                        {
                            let r = r.subst(self.infcx.tcx, substs);
                            if let ty::RegionKind::ReStatic = r {
                                found = true;
                                break;
                            } else {
                                // If there's already a lifetime bound, don't
                                // suggest anything.
                                return;
                            }
                        }
                    }

                    found
                };

                debug!(
                    "add_static_impl_trait_suggestion: has_static_predicate={:?}",
                    has_static_predicate
                );
                let static_str = kw::StaticLifetime;
                // If there is a static predicate, then the only sensible suggestion is to replace
                // fr with `'static`.
                if has_static_predicate {
                    diag.help(&format!("consider replacing `{}` with `{}`", fr_name, static_str));
                } else {
                    // Otherwise, we should suggest adding a constraint on the return type.
                    let span = self.infcx.tcx.def_span(did);
                    if let Ok(snippet) = self.infcx.tcx.sess.source_map().span_to_snippet(span) {
                        let suggestable_fr_name = if fr_name.was_named() {
                            fr_name.to_string()
                        } else {
                            "'_".to_string()
                        };
                        let span = if snippet.ends_with(';') {
                            // `type X = impl Trait;`
                            span.with_hi(span.hi() - BytePos(1))
                        } else {
                            span
                        };
                        let suggestion = format!(" + {}", suggestable_fr_name);
                        let span = span.shrink_to_hi();
                        diag.span_suggestion(
                            span,
                            &format!(
                                "to allow this `impl Trait` to capture borrowed data with lifetime \
                                 `{}`, add `{}` as a bound",
                                fr_name, suggestable_fr_name,
                            ),
                            suggestion,
                            Applicability::MachineApplicable,
                        );
                    }
                }
            }
        }
    }
}
