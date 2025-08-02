use std::ops::ControlFlow;

use rustc_errors::{Applicability, Diag, E0283, E0284, E0790, MultiSpan, struct_span_code_err};
use rustc_hir as hir;
use rustc_hir::LangItem;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CRATE_DEF_ID, DefId};
use rustc_hir::intravisit::Visitor as _;
use rustc_infer::infer::{BoundRegionConversionTime, InferCtxt};
use rustc_infer::traits::util::elaborate;
use rustc_infer::traits::{
    Obligation, ObligationCause, ObligationCauseCode, PolyTraitObligation, PredicateObligation,
};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitable as _, TypeVisitableExt as _};
use rustc_session::parse::feature_err_unstable_feature_bound;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};
use tracing::{debug, instrument};

use crate::error_reporting::TypeErrCtxt;
use crate::error_reporting::infer::need_type_info::TypeAnnotationNeeded;
use crate::error_reporting::traits::{FindExprBySpan, to_pretty_impl_header};
use crate::traits::ObligationCtxt;
use crate::traits::query::evaluate_obligation::InferCtxtExt;

#[derive(Debug)]
pub enum CandidateSource {
    DefId(DefId),
    ParamEnv(Span),
}

pub fn compute_applicable_impls_for_diagnostics<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligation: &PolyTraitObligation<'tcx>,
) -> Vec<CandidateSource> {
    let tcx = infcx.tcx;
    let param_env = obligation.param_env;

    let predicate_polarity = obligation.predicate.skip_binder().polarity;

    let impl_may_apply = |impl_def_id| {
        let ocx = ObligationCtxt::new(infcx);
        infcx.enter_forall(obligation.predicate, |placeholder_obligation| {
            let obligation_trait_ref = ocx.normalize(
                &ObligationCause::dummy(),
                param_env,
                placeholder_obligation.trait_ref,
            );

            let impl_args = infcx.fresh_args_for_item(DUMMY_SP, impl_def_id);
            let impl_trait_ref =
                tcx.impl_trait_ref(impl_def_id).unwrap().instantiate(tcx, impl_args);
            let impl_trait_ref =
                ocx.normalize(&ObligationCause::dummy(), param_env, impl_trait_ref);

            if let Err(_) =
                ocx.eq(&ObligationCause::dummy(), param_env, obligation_trait_ref, impl_trait_ref)
            {
                return false;
            }

            let impl_trait_header = tcx.impl_trait_header(impl_def_id).unwrap();
            let impl_polarity = impl_trait_header.polarity;

            match (impl_polarity, predicate_polarity) {
                (ty::ImplPolarity::Positive, ty::PredicatePolarity::Positive)
                | (ty::ImplPolarity::Negative, ty::PredicatePolarity::Negative) => {}
                _ => return false,
            }

            let obligations = tcx
                .predicates_of(impl_def_id)
                .instantiate(tcx, impl_args)
                .into_iter()
                .map(|(predicate, _)| {
                    Obligation::new(tcx, ObligationCause::dummy(), param_env, predicate)
                })
                // Kinda hacky, but let's just throw away obligations that overflow.
                // This may reduce the accuracy of this check (if the obligation guides
                // inference or it actually resulted in error after others are processed)
                // ... but this is diagnostics code.
                .filter(|obligation| {
                    infcx.next_trait_solver() || infcx.evaluate_obligation(obligation).is_ok()
                });
            ocx.register_obligations(obligations);

            ocx.select_where_possible().is_empty()
        })
    };

    let param_env_candidate_may_apply = |poly_trait_predicate: ty::PolyTraitPredicate<'tcx>| {
        let ocx = ObligationCtxt::new(infcx);
        infcx.enter_forall(obligation.predicate, |placeholder_obligation| {
            let obligation_trait_ref = ocx.normalize(
                &ObligationCause::dummy(),
                param_env,
                placeholder_obligation.trait_ref,
            );

            let param_env_predicate = infcx.instantiate_binder_with_fresh_vars(
                DUMMY_SP,
                BoundRegionConversionTime::HigherRankedType,
                poly_trait_predicate,
            );
            let param_env_trait_ref =
                ocx.normalize(&ObligationCause::dummy(), param_env, param_env_predicate.trait_ref);

            if let Err(_) = ocx.eq(
                &ObligationCause::dummy(),
                param_env,
                obligation_trait_ref,
                param_env_trait_ref,
            ) {
                return false;
            }

            ocx.select_where_possible().is_empty()
        })
    };

    let mut ambiguities = Vec::new();

    tcx.for_each_relevant_impl(
        obligation.predicate.def_id(),
        obligation.predicate.skip_binder().trait_ref.self_ty(),
        |impl_def_id| {
            if infcx.probe(|_| impl_may_apply(impl_def_id)) {
                ambiguities.push(CandidateSource::DefId(impl_def_id))
            }
        },
    );

    // If our `body_id` has been set (and isn't just from a dummy obligation cause),
    // then try to look for a param-env clause that would apply. The way we compute
    // this is somewhat manual, since we need the spans, so we elaborate this directly
    // from `predicates_of` rather than actually looking at the param-env which
    // otherwise would be more appropriate.
    let body_id = obligation.cause.body_id;
    if body_id != CRATE_DEF_ID {
        let predicates = tcx.predicates_of(body_id.to_def_id()).instantiate_identity(tcx);
        for (pred, span) in elaborate(tcx, predicates.into_iter()) {
            let kind = pred.kind();
            if let ty::ClauseKind::Trait(trait_pred) = kind.skip_binder()
                && param_env_candidate_may_apply(kind.rebind(trait_pred))
            {
                if kind.rebind(trait_pred.trait_ref)
                    == ty::Binder::dummy(ty::TraitRef::identity(tcx, trait_pred.def_id()))
                {
                    ambiguities.push(CandidateSource::ParamEnv(tcx.def_span(trait_pred.def_id())))
                } else {
                    ambiguities.push(CandidateSource::ParamEnv(span))
                }
            }
        }
    }

    ambiguities
}

impl<'a, 'tcx> TypeErrCtxt<'a, 'tcx> {
    #[instrument(skip(self), level = "debug")]
    pub(super) fn maybe_report_ambiguity(
        &self,
        obligation: &PredicateObligation<'tcx>,
    ) -> ErrorGuaranteed {
        // Unable to successfully determine, probably means
        // insufficient type information, but could mean
        // ambiguous impls. The latter *ought* to be a
        // coherence violation, so we don't report it here.

        let predicate = self.resolve_vars_if_possible(obligation.predicate);
        let span = obligation.cause.span;
        let mut long_ty_path = None;

        debug!(?predicate, obligation.cause.code = ?obligation.cause.code());

        // Ambiguity errors are often caused as fallout from earlier errors.
        // We ignore them if this `infcx` is tainted in some cases below.

        let bound_predicate = predicate.kind();
        let mut err = match bound_predicate.skip_binder() {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(data)) => {
                let trait_pred = bound_predicate.rebind(data);
                debug!(?trait_pred);

                if let Err(e) = predicate.error_reported() {
                    return e;
                }

                if let Err(guar) = self.tcx.ensure_ok().coherent_trait(trait_pred.def_id()) {
                    // Avoid bogus "type annotations needed `Foo: Bar`" errors on `impl Bar for Foo` in case
                    // other `Foo` impls are incoherent.
                    return guar;
                }

                // This is kind of a hack: it frequently happens that some earlier
                // error prevents types from being fully inferred, and then we get
                // a bunch of uninteresting errors saying something like "<generic
                // #0> doesn't implement Sized". It may even be true that we
                // could just skip over all checks where the self-ty is an
                // inference variable, but I was afraid that there might be an
                // inference variable created, registered as an obligation, and
                // then never forced by writeback, and hence by skipping here we'd
                // be ignoring the fact that we don't KNOW the type works
                // out. Though even that would probably be harmless, given that
                // we're only talking about builtin traits, which are known to be
                // inhabited. We used to check for `self.tainted_by_errors()` to
                // avoid inundating the user with unnecessary errors, but we now
                // check upstream for type errors and don't add the obligations to
                // begin with in those cases.
                if matches!(
                    self.tcx.as_lang_item(trait_pred.def_id()),
                    Some(LangItem::Sized | LangItem::MetaSized)
                ) {
                    return match self.tainted_by_errors() {
                        None => self
                            .emit_inference_failure_err(
                                obligation.cause.body_id,
                                span,
                                trait_pred.self_ty().skip_binder().into(),
                                TypeAnnotationNeeded::E0282,
                                false,
                            )
                            .emit(),
                        Some(e) => e,
                    };
                }

                // Typically, this ambiguity should only happen if
                // there are unresolved type inference variables
                // (otherwise it would suggest a coherence
                // failure). But given #21974 that is not necessarily
                // the case -- we can have multiple where clauses that
                // are only distinguished by a region, which results
                // in an ambiguity even when all types are fully
                // known, since we don't dispatch based on region
                // relationships.

                // Pick the first generic parameter that still contains inference variables as the one
                // we're going to emit an error for. If there are none (see above), fall back to
                // a more general error.
                let term = data
                    .trait_ref
                    .args
                    .iter()
                    .filter_map(ty::GenericArg::as_term)
                    .find(|s| s.has_non_region_infer());

                let mut err = if let Some(term) = term {
                    self.emit_inference_failure_err(
                        obligation.cause.body_id,
                        span,
                        term,
                        TypeAnnotationNeeded::E0283,
                        true,
                    )
                } else {
                    struct_span_code_err!(
                        self.dcx(),
                        span,
                        E0283,
                        "type annotations needed: cannot satisfy `{}`",
                        self.tcx.short_string(predicate, &mut long_ty_path),
                    )
                    .with_long_ty_path(long_ty_path)
                };

                let mut ambiguities = compute_applicable_impls_for_diagnostics(
                    self.infcx,
                    &obligation.with(self.tcx, trait_pred),
                );
                let has_non_region_infer = trait_pred
                    .skip_binder()
                    .trait_ref
                    .args
                    .types()
                    .any(|t| !t.is_ty_or_numeric_infer());
                // It doesn't make sense to talk about applicable impls if there are more than a
                // handful of them. If there are a lot of them, but only a few of them have no type
                // params, we only show those, as they are more likely to be useful/intended.
                if ambiguities.len() > 5 {
                    let infcx = self.infcx;
                    if !ambiguities.iter().all(|option| match option {
                        CandidateSource::DefId(did) => infcx.tcx.generics_of(*did).count() == 0,
                        CandidateSource::ParamEnv(_) => true,
                    }) {
                        // If not all are blanket impls, we filter blanked impls out.
                        ambiguities.retain(|option| match option {
                            CandidateSource::DefId(did) => infcx.tcx.generics_of(*did).count() == 0,
                            CandidateSource::ParamEnv(_) => true,
                        });
                    }
                }
                if ambiguities.len() > 1 && ambiguities.len() < 10 && has_non_region_infer {
                    if let Some(e) = self.tainted_by_errors()
                        && term.is_none()
                    {
                        // If `arg.is_none()`, then this is probably two param-env
                        // candidates or impl candidates that are equal modulo lifetimes.
                        // Therefore, if we've already emitted an error, just skip this
                        // one, since it's not particularly actionable.
                        err.cancel();
                        return e;
                    }
                    self.annotate_source_of_ambiguity(&mut err, &ambiguities, predicate);
                } else {
                    if let Some(e) = self.tainted_by_errors() {
                        err.cancel();
                        return e;
                    }
                    let pred = self.tcx.short_string(predicate, &mut err.long_ty_path());
                    err.note(format!("cannot satisfy `{pred}`"));
                    let impl_candidates =
                        self.find_similar_impl_candidates(predicate.as_trait_clause().unwrap());
                    if impl_candidates.len() < 40 {
                        self.report_similar_impl_candidates(
                            impl_candidates.as_slice(),
                            trait_pred,
                            obligation.cause.body_id,
                            &mut err,
                            false,
                            obligation.param_env,
                        );
                    }
                }

                if let ObligationCauseCode::WhereClause(def_id, _)
                | ObligationCauseCode::WhereClauseInExpr(def_id, ..) = *obligation.cause.code()
                {
                    self.suggest_fully_qualified_path(&mut err, def_id, span, trait_pred.def_id());
                }

                if term.is_some_and(|term| term.as_type().is_some())
                    && let Some(body) = self.tcx.hir_maybe_body_owned_by(obligation.cause.body_id)
                {
                    let mut expr_finder = FindExprBySpan::new(span, self.tcx);
                    expr_finder.visit_expr(&body.value);

                    if let Some(hir::Expr {
                        kind:
                            hir::ExprKind::Call(
                                hir::Expr {
                                    kind: hir::ExprKind::Path(hir::QPath::Resolved(None, path)),
                                    ..
                                },
                                _,
                            )
                            | hir::ExprKind::Path(hir::QPath::Resolved(None, path)),
                        ..
                    }) = expr_finder.result
                        && let [
                            ..,
                            trait_path_segment @ hir::PathSegment {
                                res: Res::Def(DefKind::Trait, trait_id),
                                ..
                            },
                            hir::PathSegment {
                                ident: assoc_item_ident,
                                res: Res::Def(_, item_id),
                                ..
                            },
                        ] = path.segments
                        && data.trait_ref.def_id == *trait_id
                        && self.tcx.trait_of_assoc(*item_id) == Some(*trait_id)
                        && let None = self.tainted_by_errors()
                    {
                        let assoc_item = self.tcx.associated_item(item_id);
                        let (verb, noun) = match assoc_item.kind {
                            ty::AssocKind::Const { .. } => ("refer to the", "constant"),
                            ty::AssocKind::Fn { .. } => ("call", "function"),
                            // This is already covered by E0223, but this following single match
                            // arm doesn't hurt here.
                            ty::AssocKind::Type { .. } => ("refer to the", "type"),
                        };

                        // Replace the more general E0283 with a more specific error
                        err.cancel();
                        err = self.dcx().struct_span_err(
                            span,
                            format!(
                                "cannot {verb} associated {noun} on trait without specifying the \
                                 corresponding `impl` type",
                            ),
                        );
                        err.code(E0790);

                        if item_id.is_local() {
                            let trait_ident = self.tcx.item_name(*trait_id);
                            err.span_label(
                                self.tcx.def_span(*item_id),
                                format!("`{trait_ident}::{assoc_item_ident}` defined here"),
                            );
                        }

                        err.span_label(span, format!("cannot {verb} associated {noun} of trait"));

                        let trait_impls = self.tcx.trait_impls_of(data.trait_ref.def_id);

                        if let Some(impl_def_id) =
                            trait_impls.non_blanket_impls().values().flatten().next()
                        {
                            let non_blanket_impl_count =
                                trait_impls.non_blanket_impls().values().flatten().count();
                            // If there is only one implementation of the trait, suggest using it.
                            // Otherwise, use a placeholder comment for the implementation.
                            let (message, self_types) = if non_blanket_impl_count == 1 {
                                (
                                    "use the fully-qualified path to the only available \
                                     implementation",
                                    vec![format!(
                                        "{}",
                                        self.tcx.type_of(impl_def_id).instantiate_identity()
                                    )],
                                )
                            } else if non_blanket_impl_count < 20 {
                                (
                                    "use a fully-qualified path to one of the available \
                                     implementations",
                                    trait_impls
                                        .non_blanket_impls()
                                        .values()
                                        .flatten()
                                        .map(|id| {
                                            format!(
                                                "{}",
                                                self.tcx.type_of(id).instantiate_identity()
                                            )
                                        })
                                        .collect::<Vec<String>>(),
                                )
                            } else {
                                (
                                    "use a fully-qualified path to a specific available \
                                     implementation",
                                    vec!["/* self type */".to_string()],
                                )
                            };
                            let suggestions: Vec<_> = self_types
                                .into_iter()
                                .map(|self_type| {
                                    let mut suggestions = vec![(
                                        path.span.shrink_to_lo(),
                                        format!("<{self_type} as "),
                                    )];
                                    if let Some(generic_arg) = trait_path_segment.args {
                                        let between_span = trait_path_segment
                                            .ident
                                            .span
                                            .between(generic_arg.span_ext);
                                        // get rid of :: between Trait and <type>
                                        // must be '::' between them, otherwise the parser won't accept the code
                                        suggestions.push((between_span, "".to_string()));
                                        suggestions.push((
                                            generic_arg.span_ext.shrink_to_hi(),
                                            ">".to_string(),
                                        ));
                                    } else {
                                        suggestions.push((
                                            trait_path_segment.ident.span.shrink_to_hi(),
                                            ">".to_string(),
                                        ));
                                    }
                                    suggestions
                                })
                                .collect();
                            err.multipart_suggestions(
                                message,
                                suggestions,
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                };

                err
            }

            ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(term)) => {
                // Same hacky approach as above to avoid deluging user
                // with error messages.

                if let Err(e) = term.error_reported() {
                    return e;
                }
                if let Some(e) = self.tainted_by_errors() {
                    return e;
                }

                self.emit_inference_failure_err(
                    obligation.cause.body_id,
                    span,
                    term,
                    TypeAnnotationNeeded::E0282,
                    false,
                )
            }

            ty::PredicateKind::Subtype(data) => {
                if let Err(e) = data.error_reported() {
                    return e;
                }
                if let Some(e) = self.tainted_by_errors() {
                    return e;
                }
                let ty::SubtypePredicate { a_is_expected: _, a, b } = data;
                // both must be type variables, or the other would've been instantiated
                assert!(a.is_ty_var() && b.is_ty_var());
                self.emit_inference_failure_err(
                    obligation.cause.body_id,
                    span,
                    a.into(),
                    TypeAnnotationNeeded::E0282,
                    true,
                )
            }

            ty::PredicateKind::Clause(ty::ClauseKind::Projection(data)) => {
                if let Err(e) = predicate.error_reported() {
                    return e;
                }
                if let Some(e) = self.tainted_by_errors() {
                    return e;
                }

                if let Err(guar) = self
                    .tcx
                    .ensure_ok()
                    .coherent_trait(self.tcx.parent(data.projection_term.def_id))
                {
                    // Avoid bogus "type annotations needed `Foo: Bar`" errors on `impl Bar for Foo` in case
                    // other `Foo` impls are incoherent.
                    return guar;
                }
                let term = data
                    .projection_term
                    .args
                    .iter()
                    .filter_map(ty::GenericArg::as_term)
                    .chain([data.term])
                    .find(|g| g.has_non_region_infer());
                let predicate = self.tcx.short_string(predicate, &mut long_ty_path);
                if let Some(term) = term {
                    self.emit_inference_failure_err(
                        obligation.cause.body_id,
                        span,
                        term,
                        TypeAnnotationNeeded::E0284,
                        true,
                    )
                    .with_note(format!("cannot satisfy `{predicate}`"))
                    .with_long_ty_path(long_ty_path)
                } else {
                    // If we can't find a generic parameter, just print a generic error
                    struct_span_code_err!(
                        self.dcx(),
                        span,
                        E0284,
                        "type annotations needed: cannot satisfy `{predicate}`",
                    )
                    .with_span_label(span, format!("cannot satisfy `{predicate}`"))
                    .with_long_ty_path(long_ty_path)
                }
            }

            ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(data)) => {
                if let Err(e) = predicate.error_reported() {
                    return e;
                }
                if let Some(e) = self.tainted_by_errors() {
                    return e;
                }
                let term =
                    data.walk().filter_map(ty::GenericArg::as_term).find(|term| term.is_infer());
                if let Some(term) = term {
                    self.emit_inference_failure_err(
                        obligation.cause.body_id,
                        span,
                        term,
                        TypeAnnotationNeeded::E0284,
                        true,
                    )
                } else {
                    // If we can't find a generic parameter, just print a generic error
                    let predicate = self.tcx.short_string(predicate, &mut long_ty_path);
                    struct_span_code_err!(
                        self.dcx(),
                        span,
                        E0284,
                        "type annotations needed: cannot satisfy `{predicate}`",
                    )
                    .with_span_label(span, format!("cannot satisfy `{predicate}`"))
                    .with_long_ty_path(long_ty_path)
                }
            }

            ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(ct, ..)) => self
                .emit_inference_failure_err(
                    obligation.cause.body_id,
                    span,
                    ct.into(),
                    TypeAnnotationNeeded::E0284,
                    true,
                ),

            ty::PredicateKind::NormalizesTo(ty::NormalizesTo { alias, term })
                if term.is_infer() =>
            {
                if let Some(e) = self.tainted_by_errors() {
                    return e;
                }
                let alias = self.tcx.short_string(alias, &mut long_ty_path);
                struct_span_code_err!(
                    self.dcx(),
                    span,
                    E0284,
                    "type annotations needed: cannot normalize `{alias}`",
                )
                .with_span_label(span, format!("cannot normalize `{alias}`"))
                .with_long_ty_path(long_ty_path)
            }

            ty::PredicateKind::Clause(ty::ClauseKind::UnstableFeature(sym)) => {
                if let Some(e) = self.tainted_by_errors() {
                    return e;
                }

                if self.tcx.features().staged_api() {
                    self.dcx().struct_span_err(
                        span,
                        format!("unstable feature `{sym}` is used without being enabled."),
                    ).with_help(format!("The feature can be enabled by marking the current item with `#[unstable_feature_bound({sym})]`"))
                } else {
                    feature_err_unstable_feature_bound(
                        &self.tcx.sess,
                        sym,
                        span,
                        format!("use of unstable library feature `{sym}`"),
                    )
                }
            }

            _ => {
                if let Some(e) = self.tainted_by_errors() {
                    return e;
                }
                let predicate = self.tcx.short_string(predicate, &mut long_ty_path);
                struct_span_code_err!(
                    self.dcx(),
                    span,
                    E0284,
                    "type annotations needed: cannot satisfy `{predicate}`",
                )
                .with_span_label(span, format!("cannot satisfy `{predicate}`"))
                .with_long_ty_path(long_ty_path)
            }
        };
        self.note_obligation_cause(&mut err, obligation);
        err.emit()
    }

    fn annotate_source_of_ambiguity(
        &self,
        err: &mut Diag<'_>,
        ambiguities: &[CandidateSource],
        predicate: ty::Predicate<'tcx>,
    ) {
        let mut spans = vec![];
        let mut crates = vec![];
        let mut post = vec![];
        let mut has_param_env = false;
        for ambiguity in ambiguities {
            match ambiguity {
                CandidateSource::DefId(impl_def_id) => match self.tcx.span_of_impl(*impl_def_id) {
                    Ok(span) => spans.push(span),
                    Err(name) => {
                        crates.push(name);
                        if let Some(header) = to_pretty_impl_header(self.tcx, *impl_def_id) {
                            post.push(header);
                        }
                    }
                },
                CandidateSource::ParamEnv(span) => {
                    has_param_env = true;
                    spans.push(*span);
                }
            }
        }
        let mut crate_names: Vec<_> = crates.iter().map(|n| format!("`{n}`")).collect();
        crate_names.sort();
        crate_names.dedup();
        post.sort();
        post.dedup();

        if self.tainted_by_errors().is_some()
            && (crate_names.len() == 1
                && spans.len() == 0
                && ["`core`", "`alloc`", "`std`"].contains(&crate_names[0].as_str())
                || predicate.visit_with(&mut HasNumericInferVisitor).is_break())
        {
            // Avoid complaining about other inference issues for expressions like
            // `42 >> 1`, where the types are still `{integer}`, but we want to
            // Do we need `trait_ref.skip_binder().self_ty().is_numeric() &&` too?
            // NOTE(eddyb) this was `.cancel()`, but `err`
            // is borrowed, so we can't fully defuse it.
            err.downgrade_to_delayed_bug();
            return;
        }

        let msg = format!(
            "multiple `impl`s{} satisfying `{}` found",
            if has_param_env { " or `where` clauses" } else { "" },
            predicate
        );
        let post = if post.len() > 1 || (post.len() == 1 && post[0].contains('\n')) {
            format!(":\n{}", post.iter().map(|p| format!("- {p}")).collect::<Vec<_>>().join("\n"),)
        } else if post.len() == 1 {
            format!(": `{}`", post[0])
        } else {
            String::new()
        };

        match (spans.len(), crates.len(), crate_names.len()) {
            (0, 0, 0) => {
                err.note(format!("cannot satisfy `{predicate}`"));
            }
            (0, _, 1) => {
                err.note(format!("{} in the `{}` crate{}", msg, crates[0], post,));
            }
            (0, _, _) => {
                err.note(format!(
                    "{} in the following crates: {}{}",
                    msg,
                    crate_names.join(", "),
                    post,
                ));
            }
            (_, 0, 0) => {
                let span: MultiSpan = spans.into();
                err.span_note(span, msg);
            }
            (_, 1, 1) => {
                let span: MultiSpan = spans.into();
                err.span_note(span, msg);
                err.note(format!("and another `impl` found in the `{}` crate{}", crates[0], post,));
            }
            _ => {
                let span: MultiSpan = spans.into();
                err.span_note(span, msg);
                err.note(format!(
                    "and more `impl`s found in the following crates: {}{}",
                    crate_names.join(", "),
                    post,
                ));
            }
        }
    }
}

struct HasNumericInferVisitor;

impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for HasNumericInferVisitor {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        if matches!(ty.kind(), ty::Infer(ty::FloatVar(_) | ty::IntVar(_))) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}
