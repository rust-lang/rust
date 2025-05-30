pub mod ambiguity;
pub mod call_kind;
mod fulfillment_errors;
pub mod on_unimplemented;
pub mod on_unimplemented_condition;
pub mod on_unimplemented_format;
mod overflow;
pub mod suggestions;

use std::{fmt, iter};

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_errors::{Applicability, Diag, E0038, E0276, MultiSpan, struct_span_code_err};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::Visitor;
use rustc_hir::{self as hir, AmbigArg};
use rustc_infer::traits::solve::Goal;
use rustc_infer::traits::{
    DynCompatibilityViolation, Obligation, ObligationCause, ObligationCauseCode,
    PredicateObligation, SelectionError,
};
use rustc_middle::ty::print::{PrintTraitRefExt as _, with_no_trimmed_paths};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::{ErrorGuaranteed, ExpnKind, Span};
use tracing::{info, instrument};

pub use self::overflow::*;
use crate::error_reporting::TypeErrCtxt;
use crate::traits::{FulfillmentError, FulfillmentErrorCode};

// When outputting impl candidates, prefer showing those that are more similar.
//
// We also compare candidates after skipping lifetimes, which has a lower
// priority than exact matches.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CandidateSimilarity {
    Exact { ignoring_lifetimes: bool },
    Fuzzy { ignoring_lifetimes: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImplCandidate<'tcx> {
    pub trait_ref: ty::TraitRef<'tcx>,
    pub similarity: CandidateSimilarity,
    impl_def_id: DefId,
}

enum GetSafeTransmuteErrorAndReason {
    Silent,
    Default,
    Error { err_msg: String, safe_transmute_explanation: Option<String> },
}

struct UnsatisfiedConst(pub bool);

/// Crude way of getting back an `Expr` from a `Span`.
pub struct FindExprBySpan<'hir> {
    pub span: Span,
    pub result: Option<&'hir hir::Expr<'hir>>,
    pub ty_result: Option<&'hir hir::Ty<'hir>>,
    pub include_closures: bool,
    pub tcx: TyCtxt<'hir>,
}

impl<'hir> FindExprBySpan<'hir> {
    pub fn new(span: Span, tcx: TyCtxt<'hir>) -> Self {
        Self { span, result: None, ty_result: None, tcx, include_closures: false }
    }
}

impl<'v> Visitor<'v> for FindExprBySpan<'v> {
    type NestedFilter = rustc_middle::hir::nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_expr(&mut self, ex: &'v hir::Expr<'v>) {
        if self.span == ex.span {
            self.result = Some(ex);
        } else {
            if let hir::ExprKind::Closure(..) = ex.kind
                && self.include_closures
                && let closure_header_sp = self.span.with_hi(ex.span.hi())
                && closure_header_sp == ex.span
            {
                self.result = Some(ex);
            }
            hir::intravisit::walk_expr(self, ex);
        }
    }

    fn visit_ty(&mut self, ty: &'v hir::Ty<'v, AmbigArg>) {
        if self.span == ty.span {
            self.ty_result = Some(ty.as_unambig_ty());
        } else {
            hir::intravisit::walk_ty(self, ty);
        }
    }
}

/// Summarizes information
#[derive(Clone)]
pub enum ArgKind {
    /// An argument of non-tuple type. Parameters are (name, ty)
    Arg(String, String),

    /// An argument of tuple type. For a "found" argument, the span is
    /// the location in the source of the pattern. For an "expected"
    /// argument, it will be None. The vector is a list of (name, ty)
    /// strings for the components of the tuple.
    Tuple(Option<Span>, Vec<(String, String)>),
}

impl ArgKind {
    fn empty() -> ArgKind {
        ArgKind::Arg("_".to_owned(), "_".to_owned())
    }

    /// Creates an `ArgKind` from the expected type of an
    /// argument. It has no name (`_`) and an optional source span.
    pub fn from_expected_ty(t: Ty<'_>, span: Option<Span>) -> ArgKind {
        match t.kind() {
            ty::Tuple(tys) => ArgKind::Tuple(
                span,
                tys.iter().map(|ty| ("_".to_owned(), ty.to_string())).collect::<Vec<_>>(),
            ),
            _ => ArgKind::Arg("_".to_owned(), t.to_string()),
        }
    }
}

#[derive(Copy, Clone)]
pub enum DefIdOrName {
    DefId(DefId),
    Name(&'static str),
}

impl<'a, 'tcx> TypeErrCtxt<'a, 'tcx> {
    pub fn report_fulfillment_errors(
        &self,
        mut errors: Vec<FulfillmentError<'tcx>>,
    ) -> ErrorGuaranteed {
        self.sub_relations
            .borrow_mut()
            .add_constraints(self, errors.iter().map(|e| e.obligation.predicate));

        #[derive(Debug)]
        struct ErrorDescriptor<'tcx> {
            goal: Goal<'tcx, ty::Predicate<'tcx>>,
            index: Option<usize>, // None if this is an old error
        }

        let mut error_map: FxIndexMap<_, Vec<_>> = self
            .reported_trait_errors
            .borrow()
            .iter()
            .map(|(&span, goals)| {
                (span, goals.0.iter().map(|&goal| ErrorDescriptor { goal, index: None }).collect())
            })
            .collect();

        // Ensure `T: Sized`, `T: MetaSized`, `T: PointeeSized` and `T: WF` obligations come last.
        // This lets us display diagnostics with more relevant type information and hide redundant
        // E0282 errors.
        errors.sort_by_key(|e| {
            let maybe_sizedness_did = match e.obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) => Some(pred.def_id()),
                ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(pred)) => Some(pred.def_id()),
                _ => None,
            };

            match e.obligation.predicate.kind().skip_binder() {
                _ if maybe_sizedness_did == self.tcx.lang_items().sized_trait() => 1,
                _ if maybe_sizedness_did == self.tcx.lang_items().meta_sized_trait() => 2,
                _ if maybe_sizedness_did == self.tcx.lang_items().pointee_sized_trait() => 3,
                ty::PredicateKind::Coerce(_) => 4,
                ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(_)) => 5,
                _ => 0,
            }
        });

        for (index, error) in errors.iter().enumerate() {
            // We want to ignore desugarings here: spans are equivalent even
            // if one is the result of a desugaring and the other is not.
            let mut span = error.obligation.cause.span;
            let expn_data = span.ctxt().outer_expn_data();
            if let ExpnKind::Desugaring(_) = expn_data.kind {
                span = expn_data.call_site;
            }

            error_map
                .entry(span)
                .or_default()
                .push(ErrorDescriptor { goal: error.obligation.as_goal(), index: Some(index) });
        }

        // We do this in 2 passes because we want to display errors in order, though
        // maybe it *is* better to sort errors by span or something.
        let mut is_suppressed = vec![false; errors.len()];
        for (_, error_set) in error_map.iter() {
            // We want to suppress "duplicate" errors with the same span.
            for error in error_set {
                if let Some(index) = error.index {
                    // Suppress errors that are either:
                    // 1) strictly implied by another error.
                    // 2) implied by an error with a smaller index.
                    for error2 in error_set {
                        if error2.index.is_some_and(|index2| is_suppressed[index2]) {
                            // Avoid errors being suppressed by already-suppressed
                            // errors, to prevent all errors from being suppressed
                            // at once.
                            continue;
                        }

                        if self.error_implies(error2.goal, error.goal)
                            && !(error2.index >= error.index
                                && self.error_implies(error.goal, error2.goal))
                        {
                            info!("skipping {:?} (implied by {:?})", error, error2);
                            is_suppressed[index] = true;
                            break;
                        }
                    }
                }
            }
        }

        let mut reported = None;

        for from_expansion in [false, true] {
            for (error, suppressed) in iter::zip(&errors, &is_suppressed) {
                if !suppressed && error.obligation.cause.span.from_expansion() == from_expansion {
                    let guar = self.report_fulfillment_error(error);
                    self.infcx.set_tainted_by_errors(guar);
                    reported = Some(guar);
                    // We want to ignore desugarings here: spans are equivalent even
                    // if one is the result of a desugaring and the other is not.
                    let mut span = error.obligation.cause.span;
                    let expn_data = span.ctxt().outer_expn_data();
                    if let ExpnKind::Desugaring(_) = expn_data.kind {
                        span = expn_data.call_site;
                    }
                    self.reported_trait_errors
                        .borrow_mut()
                        .entry(span)
                        .or_insert_with(|| (vec![], guar))
                        .0
                        .push(error.obligation.as_goal());
                }
            }
        }

        // It could be that we don't report an error because we have seen an `ErrorReported` from
        // another source. We should probably be able to fix most of these, but some are delayed
        // bugs that get a proper error after this function.
        reported.unwrap_or_else(|| self.dcx().delayed_bug("failed to report fulfillment errors"))
    }

    #[instrument(skip(self), level = "debug")]
    fn report_fulfillment_error(&self, error: &FulfillmentError<'tcx>) -> ErrorGuaranteed {
        let mut error = FulfillmentError {
            obligation: error.obligation.clone(),
            code: error.code.clone(),
            root_obligation: error.root_obligation.clone(),
        };
        if matches!(
            error.code,
            FulfillmentErrorCode::Select(crate::traits::SelectionError::Unimplemented)
                | FulfillmentErrorCode::Project(_)
        ) && self.apply_do_not_recommend(&mut error.obligation)
        {
            error.code = FulfillmentErrorCode::Select(SelectionError::Unimplemented);
        }

        match error.code {
            FulfillmentErrorCode::Select(ref selection_error) => self.report_selection_error(
                error.obligation.clone(),
                &error.root_obligation,
                selection_error,
            ),
            FulfillmentErrorCode::Project(ref e) => {
                self.report_projection_error(&error.obligation, e)
            }
            FulfillmentErrorCode::Ambiguity { overflow: None } => {
                self.maybe_report_ambiguity(&error.obligation)
            }
            FulfillmentErrorCode::Ambiguity { overflow: Some(suggest_increasing_limit) } => {
                self.report_overflow_no_abort(error.obligation.clone(), suggest_increasing_limit)
            }
            FulfillmentErrorCode::Subtype(ref expected_found, ref err) => self
                .report_mismatched_types(
                    &error.obligation.cause,
                    error.obligation.param_env,
                    expected_found.expected,
                    expected_found.found,
                    *err,
                )
                .emit(),
            FulfillmentErrorCode::ConstEquate(ref expected_found, ref err) => {
                let mut diag = self.report_mismatched_consts(
                    &error.obligation.cause,
                    error.obligation.param_env,
                    expected_found.expected,
                    expected_found.found,
                    *err,
                );
                let code = error.obligation.cause.code().peel_derives().peel_match_impls();
                if let ObligationCauseCode::WhereClause(..)
                | ObligationCauseCode::WhereClauseInExpr(..) = code
                {
                    self.note_obligation_cause_code(
                        error.obligation.cause.body_id,
                        &mut diag,
                        error.obligation.predicate,
                        error.obligation.param_env,
                        code,
                        &mut vec![],
                        &mut Default::default(),
                    );
                }
                diag.emit()
            }
            FulfillmentErrorCode::Cycle(ref cycle) => self.report_overflow_obligation_cycle(cycle),
        }
    }
}

/// Recovers the "impl X for Y" signature from `impl_def_id` and returns it as a
/// string.
pub(crate) fn to_pretty_impl_header(tcx: TyCtxt<'_>, impl_def_id: DefId) -> Option<String> {
    use std::fmt::Write;

    let trait_ref = tcx.impl_trait_ref(impl_def_id)?.instantiate_identity();
    let mut w = "impl".to_owned();

    #[derive(Debug, Default)]
    struct SizednessFound {
        sized: bool,
        meta_sized: bool,
    }

    let mut types_with_sizedness_bounds = FxIndexMap::<_, SizednessFound>::default();

    let args = ty::GenericArgs::identity_for_item(tcx, impl_def_id);

    let arg_names = args.iter().map(|k| k.to_string()).filter(|k| k != "'_").collect::<Vec<_>>();
    if !arg_names.is_empty() {
        w.push('<');
        w.push_str(&arg_names.join(", "));
        w.push('>');

        for ty in args.types() {
            // `PointeeSized` params might have no predicates.
            types_with_sizedness_bounds.insert(ty, SizednessFound::default());
        }
    }

    write!(
        w,
        " {}{} for {}",
        tcx.impl_polarity(impl_def_id).as_str(),
        trait_ref.print_only_trait_path(),
        tcx.type_of(impl_def_id).instantiate_identity()
    )
    .unwrap();

    let predicates = tcx.predicates_of(impl_def_id).predicates;
    let mut pretty_predicates = Vec::with_capacity(predicates.len());

    let sized_trait = tcx.lang_items().sized_trait();
    let meta_sized_trait = tcx.lang_items().meta_sized_trait();

    for (p, _) in predicates {
        // Accumulate the sizedness bounds for each self ty.
        if let Some(trait_clause) = p.as_trait_clause() {
            let self_ty = trait_clause.self_ty().skip_binder();
            let sizedness_of = types_with_sizedness_bounds.entry(self_ty).or_default();
            if Some(trait_clause.def_id()) == sized_trait {
                sizedness_of.sized = true;
                continue;
            } else if Some(trait_clause.def_id()) == meta_sized_trait {
                sizedness_of.meta_sized = true;
                continue;
            }
        }

        pretty_predicates.push(p.to_string());
    }

    for (ty, sizedness) in types_with_sizedness_bounds {
        if !tcx.features().sized_hierarchy() {
            if sizedness.sized {
                // Maybe a default bound, don't write anything.
            } else {
                pretty_predicates.push(format!("{ty}: ?Sized"));
            }
        } else {
            if sizedness.sized {
                // Maybe a default bound, don't write anything.
                pretty_predicates.push(format!("{ty}: Sized"));
            } else if sizedness.meta_sized {
                pretty_predicates.push(format!("{ty}: MetaSized"));
            } else {
                pretty_predicates.push(format!("{ty}: PointeeSized"));
            }
        }
    }

    if !pretty_predicates.is_empty() {
        write!(w, "\n  where {}", pretty_predicates.join(", ")).unwrap();
    }

    w.push(';');
    Some(w)
}

impl<'a, 'tcx> TypeErrCtxt<'a, 'tcx> {
    pub fn report_extra_impl_obligation(
        &self,
        error_span: Span,
        impl_item_def_id: LocalDefId,
        trait_item_def_id: DefId,
        requirement: &dyn fmt::Display,
    ) -> Diag<'a> {
        let mut err = struct_span_code_err!(
            self.dcx(),
            error_span,
            E0276,
            "impl has stricter requirements than trait"
        );

        if !self.tcx.is_impl_trait_in_trait(trait_item_def_id) {
            if let Some(span) = self.tcx.hir_span_if_local(trait_item_def_id) {
                let item_name = self.tcx.item_name(impl_item_def_id.to_def_id());
                err.span_label(span, format!("definition of `{item_name}` from trait"));
            }
        }

        err.span_label(error_span, format!("impl has extra requirement {requirement}"));

        err
    }
}

pub fn report_dyn_incompatibility<'tcx>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    hir_id: Option<hir::HirId>,
    trait_def_id: DefId,
    violations: &[DynCompatibilityViolation],
) -> Diag<'tcx> {
    let trait_str = tcx.def_path_str(trait_def_id);
    let trait_span = tcx.hir_get_if_local(trait_def_id).and_then(|node| match node {
        hir::Node::Item(item) => match item.kind {
            hir::ItemKind::Trait(_, _, ident, ..) | hir::ItemKind::TraitAlias(ident, _, _) => {
                Some(ident.span)
            }
            _ => unreachable!(),
        },
        _ => None,
    });

    let mut err = struct_span_code_err!(
        tcx.dcx(),
        span,
        E0038,
        "the {} `{}` is not dyn compatible",
        tcx.def_descr(trait_def_id),
        trait_str
    );
    err.span_label(span, format!("`{trait_str}` is not dyn compatible"));

    attempt_dyn_to_impl_suggestion(tcx, hir_id, &mut err);

    let mut reported_violations = FxIndexSet::default();
    let mut multi_span = vec![];
    let mut messages = vec![];
    for violation in violations {
        if let DynCompatibilityViolation::SizedSelf(sp) = &violation
            && !sp.is_empty()
        {
            // Do not report `SizedSelf` without spans pointing at `SizedSelf` obligations
            // with a `Span`.
            reported_violations.insert(DynCompatibilityViolation::SizedSelf(vec![].into()));
        }
        if reported_violations.insert(violation.clone()) {
            let spans = violation.spans();
            let msg = if trait_span.is_none() || spans.is_empty() {
                format!("the trait is not dyn compatible because {}", violation.error_msg())
            } else {
                format!("...because {}", violation.error_msg())
            };
            if spans.is_empty() {
                err.note(msg);
            } else {
                for span in spans {
                    multi_span.push(span);
                    messages.push(msg.clone());
                }
            }
        }
    }
    let has_multi_span = !multi_span.is_empty();
    let mut note_span = MultiSpan::from_spans(multi_span.clone());
    if let (Some(trait_span), true) = (trait_span, has_multi_span) {
        note_span.push_span_label(trait_span, "this trait is not dyn compatible...");
    }
    for (span, msg) in iter::zip(multi_span, messages) {
        note_span.push_span_label(span, msg);
    }
    err.span_note(
        note_span,
        "for a trait to be dyn compatible it needs to allow building a vtable\n\
        for more information, visit <https://doc.rust-lang.org/reference/items/traits.html#dyn-compatibility>",
    );

    // Only provide the help if its a local trait, otherwise it's not actionable.
    if trait_span.is_some() {
        let mut potential_solutions: Vec<_> =
            reported_violations.into_iter().map(|violation| violation.solution()).collect();
        potential_solutions.sort();
        // Allows us to skip suggesting that the same item should be moved to another trait multiple times.
        potential_solutions.dedup();
        for solution in potential_solutions {
            solution.add_to(&mut err);
        }
    }

    attempt_dyn_to_enum_suggestion(tcx, trait_def_id, &*trait_str, &mut err);

    err
}

/// Attempt to suggest converting the `dyn Trait` argument to an enumeration
/// over the types that implement `Trait`.
fn attempt_dyn_to_enum_suggestion(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
    trait_str: &str,
    err: &mut Diag<'_>,
) {
    let impls_of = tcx.trait_impls_of(trait_def_id);

    if !impls_of.blanket_impls().is_empty() {
        return;
    }

    let concrete_impls: Option<Vec<Ty<'_>>> = impls_of
        .non_blanket_impls()
        .values()
        .flatten()
        .map(|impl_id| {
            // Don't suggest conversion to enum if the impl types have type parameters.
            // It's unlikely the user wants to define a generic enum.
            let Some(impl_type) = tcx.type_of(*impl_id).no_bound_vars() else { return None };

            // Obviously unsized impl types won't be usable in an enum.
            // Note: this doesn't use `Ty::has_trivial_sizedness` because that function
            // defaults to assuming that things are *not* sized, whereas we want to
            // fall back to assuming that things may be sized.
            match impl_type.kind() {
                ty::Str | ty::Slice(_) | ty::Dynamic(_, _, ty::DynKind::Dyn) => {
                    return None;
                }
                _ => {}
            }
            Some(impl_type)
        })
        .collect();
    let Some(concrete_impls) = concrete_impls else { return };

    const MAX_IMPLS_TO_SUGGEST_CONVERTING_TO_ENUM: usize = 9;
    if concrete_impls.is_empty() || concrete_impls.len() > MAX_IMPLS_TO_SUGGEST_CONVERTING_TO_ENUM {
        return;
    }

    let externally_visible = if let Some(def_id) = trait_def_id.as_local() {
        // We may be executing this during typeck, which would result in cycle
        // if we used effective_visibilities query, which looks into opaque types
        // (and therefore calls typeck).
        tcx.resolutions(()).effective_visibilities.is_exported(def_id)
    } else {
        false
    };

    if let [only_impl] = &concrete_impls[..] {
        let within = if externally_visible { " within this crate" } else { "" };
        err.help(with_no_trimmed_paths!(format!(
            "only type `{only_impl}` implements `{trait_str}`{within}; \
            consider using it directly instead."
        )));
    } else {
        let types = concrete_impls
            .iter()
            .map(|t| with_no_trimmed_paths!(format!("  {}", t)))
            .collect::<Vec<String>>()
            .join("\n");

        err.help(format!(
            "the following types implement `{trait_str}`:\n\
             {types}\n\
             consider defining an enum where each variant holds one of these types,\n\
             implementing `{trait_str}` for this new enum and using it instead",
        ));
    }

    if externally_visible {
        err.note(format!(
            "`{trait_str}` may be implemented in other crates; if you want to support your users \
             passing their own types here, you can't refer to a specific type",
        ));
    }
}

/// Attempt to suggest that a `dyn Trait` argument or return type be converted
/// to use `impl Trait`.
fn attempt_dyn_to_impl_suggestion(tcx: TyCtxt<'_>, hir_id: Option<hir::HirId>, err: &mut Diag<'_>) {
    let Some(hir_id) = hir_id else { return };
    let hir::Node::Ty(ty) = tcx.hir_node(hir_id) else { return };
    let hir::TyKind::TraitObject([trait_ref, ..], ..) = ty.kind else { return };

    // Only suggest converting `dyn` to `impl` if we're in a function signature.
    // This ensures that we don't suggest converting e.g.
    //   `type Alias = Box<dyn DynIncompatibleTrait>;` to
    //   `type Alias = Box<impl DynIncompatibleTrait>;`
    let Some((_id, first_non_type_parent_node)) =
        tcx.hir_parent_iter(hir_id).find(|(_id, node)| !matches!(node, hir::Node::Ty(_)))
    else {
        return;
    };
    if first_non_type_parent_node.fn_sig().is_none() {
        return;
    }

    err.span_suggestion_verbose(
        ty.span.until(trait_ref.span),
        "consider using an opaque type instead",
        "impl ",
        Applicability::MaybeIncorrect,
    );
}
