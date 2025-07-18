//! Print diagnostics to explain why values are borrowed.

#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use std::assert_matches::assert_matches;

use rustc_errors::{Applicability, Diag, EmissionGuarantee};
use rustc_hir as hir;
use rustc_hir::intravisit::Visitor;
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_middle::middle::resolve_bound_vars::ObjectLifetimeDefault;
use rustc_middle::mir::{
    Body, CallSource, CastKind, ConstraintCategory, FakeReadCause, Local, LocalInfo, Location,
    Operand, Place, Rvalue, Statement, StatementKind, TerminatorKind,
};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{self, RegionVid, Ty, TyCtxt};
use rustc_span::{DesugaringKind, Span, kw, sym};
use rustc_trait_selection::error_reporting::traits::FindExprBySpan;
use rustc_trait_selection::error_reporting::traits::call_kind::CallKind;
use tracing::{debug, instrument};

use super::{RegionName, UseSpans, find_use};
use crate::borrow_set::BorrowData;
use crate::constraints::OutlivesConstraint;
use crate::nll::ConstraintDescription;
use crate::region_infer::{BlameConstraint, Cause};
use crate::{MirBorrowckCtxt, WriteKind};

#[derive(Debug)]
pub(crate) enum BorrowExplanation<'tcx> {
    UsedLater(Local, LaterUseKind, Span, Option<Span>),
    UsedLaterInLoop(LaterUseKind, Span, Option<Span>),
    UsedLaterWhenDropped {
        drop_loc: Location,
        dropped_local: Local,
        should_note_order: bool,
    },
    MustBeValidFor {
        category: ConstraintCategory<'tcx>,
        from_closure: bool,
        span: Span,
        region_name: RegionName,
        opt_place_desc: Option<String>,
        path: Vec<OutlivesConstraint<'tcx>>,
    },
    Unexplained,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum LaterUseKind {
    TraitCapture,
    ClosureCapture,
    Call,
    FakeLetRead,
    Other,
}

impl<'tcx> BorrowExplanation<'tcx> {
    pub(crate) fn is_explained(&self) -> bool {
        !matches!(self, BorrowExplanation::Unexplained)
    }
    pub(crate) fn add_explanation_to_diagnostic<G: EmissionGuarantee>(
        &self,
        cx: &MirBorrowckCtxt<'_, '_, 'tcx>,
        err: &mut Diag<'_, G>,
        borrow_desc: &str,
        borrow_span: Option<Span>,
        multiple_borrow_span: Option<(Span, Span)>,
    ) {
        let tcx = cx.infcx.tcx;
        let body = cx.body;

        if let Some(span) = borrow_span {
            let def_id = body.source.def_id();
            if let Some(node) = tcx.hir_get_if_local(def_id)
                && let Some(body_id) = node.body_id()
            {
                let body = tcx.hir_body(body_id);
                let mut expr_finder = FindExprBySpan::new(span, tcx);
                expr_finder.visit_expr(body.value);
                if let Some(mut expr) = expr_finder.result {
                    while let hir::ExprKind::AddrOf(_, _, inner)
                    | hir::ExprKind::Unary(hir::UnOp::Deref, inner)
                    | hir::ExprKind::Field(inner, _)
                    | hir::ExprKind::MethodCall(_, inner, _, _)
                    | hir::ExprKind::Index(inner, _, _) = &expr.kind
                    {
                        expr = inner;
                    }
                    if let hir::ExprKind::Path(hir::QPath::Resolved(None, p)) = expr.kind
                        && let [hir::PathSegment { ident, args: None, .. }] = p.segments
                        && let hir::def::Res::Local(hir_id) = p.res
                        && let hir::Node::Pat(pat) = tcx.hir_node(hir_id)
                    {
                        if !ident.span.in_external_macro(tcx.sess.source_map()) {
                            err.span_label(pat.span, format!("binding `{ident}` declared here"));
                        }
                    }
                }
            }
        }
        match *self {
            BorrowExplanation::UsedLater(
                dropped_local,
                later_use_kind,
                var_or_use_span,
                path_span,
            ) => {
                let message = match later_use_kind {
                    LaterUseKind::TraitCapture => "captured here by trait object",
                    LaterUseKind::ClosureCapture => "captured here by closure",
                    LaterUseKind::Call => "used by call",
                    LaterUseKind::FakeLetRead => "stored here",
                    LaterUseKind::Other => "used here",
                };
                let local_decl = &body.local_decls[dropped_local];

                if let &LocalInfo::IfThenRescopeTemp { if_then } = local_decl.local_info()
                    && let Some((_, hir::Node::Expr(expr))) = tcx.hir_parent_iter(if_then).next()
                    && let hir::ExprKind::If(cond, conseq, alt) = expr.kind
                    && let hir::ExprKind::Let(&hir::LetExpr {
                        span: _,
                        pat,
                        init,
                        // FIXME(#101728): enable rewrite when type ascription is stabilized again
                        ty: None,
                        recovered: _,
                    }) = cond.kind
                    && pat.span.can_be_used_for_suggestions()
                    && let Ok(pat) = tcx.sess.source_map().span_to_snippet(pat.span)
                {
                    suggest_rewrite_if_let(tcx, expr, &pat, init, conseq, alt, err);
                } else if path_span.is_none_or(|path_span| path_span == var_or_use_span) {
                    // We can use `var_or_use_span` if either `path_span` is not present, or both
                    // spans are the same.
                    if borrow_span.is_none_or(|sp| !sp.overlaps(var_or_use_span)) {
                        err.span_label(
                            var_or_use_span,
                            format!("{borrow_desc}borrow later {message}"),
                        );
                    }
                } else {
                    // path_span must be `Some` as otherwise the if condition is true
                    let path_span = path_span.unwrap();
                    // path_span is only present in the case of closure capture
                    assert_matches!(later_use_kind, LaterUseKind::ClosureCapture);
                    if !borrow_span.is_some_and(|sp| sp.overlaps(var_or_use_span)) {
                        let path_label = "used here by closure";
                        let capture_kind_label = message;
                        err.span_label(
                            var_or_use_span,
                            format!("{borrow_desc}borrow later {capture_kind_label}"),
                        );
                        err.span_label(path_span, path_label);
                    }
                }
            }
            BorrowExplanation::UsedLaterInLoop(later_use_kind, var_or_use_span, path_span) => {
                let message = match later_use_kind {
                    LaterUseKind::TraitCapture => {
                        "borrow captured here by trait object, in later iteration of loop"
                    }
                    LaterUseKind::ClosureCapture => {
                        "borrow captured here by closure, in later iteration of loop"
                    }
                    LaterUseKind::Call => "borrow used by call, in later iteration of loop",
                    LaterUseKind::FakeLetRead => "borrow later stored here",
                    LaterUseKind::Other => "borrow used here, in later iteration of loop",
                };
                // We can use `var_or_use_span` if either `path_span` is not present, or both spans
                // are the same.
                if path_span.map(|path_span| path_span == var_or_use_span).unwrap_or(true) {
                    err.span_label(var_or_use_span, format!("{borrow_desc}{message}"));
                } else {
                    // path_span must be `Some` as otherwise the if condition is true
                    let path_span = path_span.unwrap();
                    // path_span is only present in the case of closure capture
                    assert_matches!(later_use_kind, LaterUseKind::ClosureCapture);
                    if borrow_span.map(|sp| !sp.overlaps(var_or_use_span)).unwrap_or(true) {
                        let path_label = "used here by closure";
                        let capture_kind_label = message;
                        err.span_label(
                            var_or_use_span,
                            format!("{borrow_desc}borrow later {capture_kind_label}"),
                        );
                        err.span_label(path_span, path_label);
                    }
                }
            }
            BorrowExplanation::UsedLaterWhenDropped {
                drop_loc,
                dropped_local,
                should_note_order,
            } => {
                let local_decl = &body.local_decls[dropped_local];
                let mut ty = local_decl.ty;
                if local_decl.source_info.span.desugaring_kind() == Some(DesugaringKind::ForLoop) {
                    if let ty::Adt(adt, args) = local_decl.ty.kind() {
                        if tcx.is_diagnostic_item(sym::Option, adt.did()) {
                            // in for loop desugaring, only look at the `Some(..)` inner type
                            ty = args.type_at(0);
                        }
                    }
                }
                let (dtor_desc, type_desc) = match ty.kind() {
                    // If type is an ADT that implements Drop, then
                    // simplify output by reporting just the ADT name.
                    ty::Adt(adt, _args) if adt.has_dtor(tcx) && !adt.is_box() => {
                        ("`Drop` code", format!("type `{}`", tcx.def_path_str(adt.did())))
                    }

                    // Otherwise, just report the whole type (and use
                    // the intentionally fuzzy phrase "destructor")
                    ty::Closure(..) => ("destructor", "closure".to_owned()),
                    ty::Coroutine(..) => ("destructor", "coroutine".to_owned()),

                    _ => ("destructor", format!("type `{}`", local_decl.ty)),
                };

                match cx.local_name(dropped_local) {
                    Some(local_name) if !local_decl.from_compiler_desugaring() => {
                        let message = format!(
                            "{borrow_desc}borrow might be used here, when `{local_name}` is dropped \
                             and runs the {dtor_desc} for {type_desc}",
                        );
                        err.span_label(body.source_info(drop_loc).span, message);

                        if should_note_order {
                            err.note(
                                "values in a scope are dropped \
                                 in the opposite order they are defined",
                            );
                        }
                    }
                    _ => {
                        err.span_label(
                            local_decl.source_info.span,
                            format!(
                                "a temporary with access to the {borrow_desc}borrow \
                                 is created here ...",
                            ),
                        );
                        let message = format!(
                            "... and the {borrow_desc}borrow might be used here, \
                             when that temporary is dropped \
                             and runs the {dtor_desc} for {type_desc}",
                        );
                        err.span_label(body.source_info(drop_loc).span, message);

                        struct FindLetExpr<'hir> {
                            span: Span,
                            result: Option<(Span, &'hir hir::Pat<'hir>, &'hir hir::Expr<'hir>)>,
                            tcx: TyCtxt<'hir>,
                        }

                        impl<'hir> rustc_hir::intravisit::Visitor<'hir> for FindLetExpr<'hir> {
                            type NestedFilter = rustc_middle::hir::nested_filter::OnlyBodies;
                            fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
                                self.tcx
                            }
                            fn visit_expr(&mut self, expr: &'hir hir::Expr<'hir>) {
                                if let hir::ExprKind::If(cond, _conseq, _alt)
                                | hir::ExprKind::Loop(
                                    &hir::Block {
                                        expr:
                                            Some(&hir::Expr {
                                                kind: hir::ExprKind::If(cond, _conseq, _alt),
                                                ..
                                            }),
                                        ..
                                    },
                                    _,
                                    hir::LoopSource::While,
                                    _,
                                ) = expr.kind
                                    && let hir::ExprKind::Let(hir::LetExpr {
                                        init: let_expr_init,
                                        span: let_expr_span,
                                        pat: let_expr_pat,
                                        ..
                                    }) = cond.kind
                                    && let_expr_init.span.contains(self.span)
                                {
                                    self.result =
                                        Some((*let_expr_span, let_expr_pat, let_expr_init))
                                } else {
                                    hir::intravisit::walk_expr(self, expr);
                                }
                            }
                        }

                        if let &LocalInfo::IfThenRescopeTemp { if_then } = local_decl.local_info()
                            && let hir::Node::Expr(expr) = tcx.hir_node(if_then)
                            && let hir::ExprKind::If(cond, conseq, alt) = expr.kind
                            && let hir::ExprKind::Let(&hir::LetExpr {
                                span: _,
                                pat,
                                init,
                                // FIXME(#101728): enable rewrite when type ascription is
                                // stabilized again.
                                ty: None,
                                recovered: _,
                            }) = cond.kind
                            && pat.span.can_be_used_for_suggestions()
                            && let Ok(pat) = tcx.sess.source_map().span_to_snippet(pat.span)
                        {
                            suggest_rewrite_if_let(tcx, expr, &pat, init, conseq, alt, err);
                        } else if let Some((old, new)) = multiple_borrow_span
                            && let def_id = body.source.def_id()
                            && let Some(node) = tcx.hir_get_if_local(def_id)
                            && let Some(body_id) = node.body_id()
                            && let hir_body = tcx.hir_body(body_id)
                            && let mut expr_finder = (FindLetExpr { span: old, result: None, tcx })
                            && let Some((let_expr_span, let_expr_pat, let_expr_init)) = {
                                expr_finder.visit_expr(hir_body.value);
                                expr_finder.result
                            }
                            && !let_expr_span.contains(new)
                        {
                            // #133941: The `old` expression is at the conditional part of an
                            // if/while let expression. Adding a semicolon won't work.
                            // Instead, try suggesting the `matches!` macro or a temporary.
                            if let_expr_pat
                                .walk_short(|pat| !matches!(pat.kind, hir::PatKind::Binding(..)))
                            {
                                if let Ok(pat_snippet) =
                                    tcx.sess.source_map().span_to_snippet(let_expr_pat.span)
                                    && let Ok(init_snippet) =
                                        tcx.sess.source_map().span_to_snippet(let_expr_init.span)
                                {
                                    err.span_suggestion_verbose(
                                        let_expr_span,
                                        "consider using the `matches!` macro",
                                        format!("matches!({init_snippet}, {pat_snippet})"),
                                        Applicability::MaybeIncorrect,
                                    );
                                } else {
                                    err.note("consider using the `matches!` macro");
                                }
                            }
                        } else if let LocalInfo::BlockTailTemp(info) = local_decl.local_info() {
                            let sp = info.span.find_oldest_ancestor_in_same_ctxt();
                            if info.tail_result_is_ignored {
                                // #85581: If the first mutable borrow's scope contains
                                // the second borrow, this suggestion isn't helpful.
                                if !multiple_borrow_span.is_some_and(|(old, new)| {
                                    old.to(info.span.shrink_to_hi()).contains(new)
                                }) {
                                    err.span_suggestion_verbose(
                                        sp.shrink_to_hi(),
                                        "consider adding semicolon after the expression so its \
                                        temporaries are dropped sooner, before the local variables \
                                        declared by the block are dropped",
                                        ";",
                                        Applicability::MaybeIncorrect,
                                    );
                                }
                            } else {
                                err.note(
                                    "the temporary is part of an expression at the end of a \
                                     block;\nconsider forcing this temporary to be dropped sooner, \
                                     before the block's local variables are dropped",
                                );
                                err.multipart_suggestion(
                                    "for example, you could save the expression's value in a new \
                                     local variable `x` and then make `x` be the expression at the \
                                     end of the block",
                                    vec![
                                        (sp.shrink_to_lo(), "let x = ".to_string()),
                                        (sp.shrink_to_hi(), "; x".to_string()),
                                    ],
                                    Applicability::MaybeIncorrect,
                                );
                            };
                        }
                    }
                }
            }
            BorrowExplanation::MustBeValidFor {
                category,
                span,
                ref region_name,
                ref opt_place_desc,
                from_closure: _,
                ref path,
            } => {
                region_name.highlight_region_name(err);

                if let Some(desc) = opt_place_desc {
                    err.span_label(
                        span,
                        format!(
                            "{}requires that `{desc}` is borrowed for `{region_name}`",
                            category.description(),
                        ),
                    );
                } else {
                    err.span_label(
                        span,
                        format!(
                            "{}requires that {borrow_desc}borrow lasts for `{region_name}`",
                            category.description(),
                        ),
                    );
                };

                cx.add_placeholder_from_predicate_note(err, &path);
                cx.add_sized_or_copy_bound_info(err, category, &path);

                if let ConstraintCategory::Cast {
                    is_implicit_coercion: true,
                    unsize_to: Some(unsize_ty),
                } = category
                {
                    self.add_object_lifetime_default_note(tcx, err, unsize_ty);
                }
                self.add_lifetime_bound_suggestion_to_diagnostic(err, &category, span, region_name);
            }
            _ => {}
        }
    }

    fn add_object_lifetime_default_note<G: EmissionGuarantee>(
        &self,
        tcx: TyCtxt<'tcx>,
        err: &mut Diag<'_, G>,
        unsize_ty: Ty<'tcx>,
    ) {
        if let ty::Adt(def, args) = unsize_ty.kind() {
            // We try to elaborate the object lifetime defaults and present those to the user. This
            // should make it clear where the region constraint is coming from.
            let generics = tcx.generics_of(def.did());

            let mut has_dyn = false;
            let mut failed = false;

            let elaborated_args =
                std::iter::zip(*args, &generics.own_params).map(|(arg, param)| {
                    if let Some(ty::Dynamic(obj, _, ty::Dyn)) = arg.as_type().map(Ty::kind) {
                        let default = tcx.object_lifetime_default(param.def_id);

                        let re_static = tcx.lifetimes.re_static;

                        let implied_region = match default {
                            // This is not entirely precise.
                            ObjectLifetimeDefault::Empty => re_static,
                            ObjectLifetimeDefault::Ambiguous => {
                                failed = true;
                                re_static
                            }
                            ObjectLifetimeDefault::Param(param_def_id) => {
                                let index = generics.param_def_id_to_index[&param_def_id] as usize;
                                args.get(index).and_then(|arg| arg.as_region()).unwrap_or_else(
                                    || {
                                        failed = true;
                                        re_static
                                    },
                                )
                            }
                            ObjectLifetimeDefault::Static => re_static,
                        };

                        has_dyn = true;

                        Ty::new_dynamic(tcx, obj, implied_region, ty::Dyn).into()
                    } else {
                        arg
                    }
                });
            let elaborated_ty = Ty::new_adt(tcx, *def, tcx.mk_args_from_iter(elaborated_args));

            if has_dyn && !failed {
                err.note(format!(
                    "due to object lifetime defaults, `{unsize_ty}` actually means `{elaborated_ty}`"
                ));
            }
        }
    }

    fn add_lifetime_bound_suggestion_to_diagnostic<G: EmissionGuarantee>(
        &self,
        err: &mut Diag<'_, G>,
        category: &ConstraintCategory<'tcx>,
        span: Span,
        region_name: &RegionName,
    ) {
        if !span.is_desugaring(DesugaringKind::OpaqueTy) {
            return;
        }
        if let ConstraintCategory::OpaqueType = category {
            let suggestable_name =
                if region_name.was_named() { region_name.name } else { kw::UnderscoreLifetime };

            let msg = format!(
                "you can add a bound to the {}to make it last less than `'static` and match `{region_name}`",
                category.description(),
            );

            err.span_suggestion_verbose(
                span.shrink_to_hi(),
                msg,
                format!(" + {suggestable_name}"),
                Applicability::Unspecified,
            );
        }
    }
}

fn suggest_rewrite_if_let<G: EmissionGuarantee>(
    tcx: TyCtxt<'_>,
    expr: &hir::Expr<'_>,
    pat: &str,
    init: &hir::Expr<'_>,
    conseq: &hir::Expr<'_>,
    alt: Option<&hir::Expr<'_>>,
    err: &mut Diag<'_, G>,
) {
    let source_map = tcx.sess.source_map();
    err.span_note(
        source_map.end_point(conseq.span),
        "lifetimes for temporaries generated in `if let`s have been shortened in Edition 2024 so that they are dropped here instead",
    );
    if expr.span.can_be_used_for_suggestions() && conseq.span.can_be_used_for_suggestions() {
        let needs_block = if let Some(hir::Node::Expr(expr)) =
            alt.and_then(|alt| tcx.hir_parent_iter(alt.hir_id).next()).map(|(_, node)| node)
        {
            matches!(expr.kind, hir::ExprKind::If(..))
        } else {
            false
        };
        let mut sugg = vec![
            (
                expr.span.shrink_to_lo().between(init.span),
                if needs_block { "{ match ".into() } else { "match ".into() },
            ),
            (conseq.span.shrink_to_lo(), format!(" {{ {pat} => ")),
        ];
        let expr_end = expr.span.shrink_to_hi();
        let mut expr_end_code;
        if let Some(alt) = alt {
            sugg.push((conseq.span.between(alt.span), " _ => ".into()));
            expr_end_code = "}".to_string();
        } else {
            expr_end_code = " _ => {} }".into();
        }
        expr_end_code.push('}');
        sugg.push((expr_end, expr_end_code));
        err.multipart_suggestion(
            "consider rewriting the `if` into `match` which preserves the extended lifetime",
            sugg,
            Applicability::MaybeIncorrect,
        );
    }
}

impl<'tcx> MirBorrowckCtxt<'_, '_, 'tcx> {
    fn free_region_constraint_info(
        &self,
        borrow_region: RegionVid,
        outlived_region: RegionVid,
    ) -> (ConstraintCategory<'tcx>, bool, Span, Option<RegionName>, Vec<OutlivesConstraint<'tcx>>)
    {
        let (blame_constraint, path) = self.regioncx.best_blame_constraint(
            borrow_region,
            NllRegionVariableOrigin::FreeRegion,
            |r| self.regioncx.provides_universal_region(r, borrow_region, outlived_region),
        );
        let BlameConstraint { category, from_closure, cause, .. } = blame_constraint;

        let outlived_fr_name = self.give_region_a_name(outlived_region);

        (category, from_closure, cause.span, outlived_fr_name, path)
    }

    /// Returns structured explanation for *why* the borrow contains the
    /// point from `location`. This is key for the "3-point errors"
    /// [described in the NLL RFC][d].
    ///
    /// # Parameters
    ///
    /// - `borrow`: the borrow in question
    /// - `location`: where the borrow occurs
    /// - `kind_place`: if Some, this describes the statement that triggered the error.
    ///   - first half is the kind of write, if any, being performed
    ///   - second half is the place being accessed
    ///
    /// [d]: https://rust-lang.github.io/rfcs/2094-nll.html#leveraging-intuition-framing-errors-in-terms-of-points
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn explain_why_borrow_contains_point(
        &self,
        location: Location,
        borrow: &BorrowData<'tcx>,
        kind_place: Option<(WriteKind, Place<'tcx>)>,
    ) -> BorrowExplanation<'tcx> {
        let regioncx = &self.regioncx;
        let body: &Body<'_> = self.body;
        let tcx = self.infcx.tcx;

        let borrow_region_vid = borrow.region;
        debug!(?borrow_region_vid);

        let mut region_sub = self.regioncx.find_sub_region_live_at(borrow_region_vid, location);
        debug!(?region_sub);

        let mut use_location = location;
        let mut use_in_later_iteration_of_loop = false;

        if region_sub == borrow_region_vid {
            // When `region_sub` is the same as `borrow_region_vid` (the location where the borrow
            // is issued is the same location that invalidates the reference), this is likely a
            // loop iteration. In this case, try using the loop terminator location in
            // `find_sub_region_live_at`.
            if let Some(loop_terminator_location) =
                regioncx.find_loop_terminator_location(borrow.region, body)
            {
                region_sub = self
                    .regioncx
                    .find_sub_region_live_at(borrow_region_vid, loop_terminator_location);
                debug!("explain_why_borrow_contains_point: region_sub in loop={:?}", region_sub);
                use_location = loop_terminator_location;
                use_in_later_iteration_of_loop = true;
            }
        }

        // NLL doesn't consider boring locals for liveness, and wouldn't encounter a
        // `Cause::LiveVar` for such a local. Polonius can't avoid computing liveness for boring
        // locals yet, and will encounter them when trying to explain why a borrow contains a given
        // point.
        //
        // We want to focus on relevant live locals in diagnostics, so when polonius is enabled, we
        // ensure that we don't emit live boring locals as explanations.
        let is_local_boring = |local| {
            if let Some(polonius_diagnostics) = self.polonius_diagnostics {
                polonius_diagnostics.boring_nll_locals.contains(&local)
            } else {
                assert!(!tcx.sess.opts.unstable_opts.polonius.is_next_enabled());

                // Boring locals are never the cause of a borrow explanation in NLLs.
                false
            }
        };
        match find_use::find(body, regioncx, tcx, region_sub, use_location) {
            Some(Cause::LiveVar(local, location)) if !is_local_boring(local) => {
                let span = body.source_info(location).span;
                let spans = self
                    .move_spans(Place::from(local).as_ref(), location)
                    .or_else(|| self.borrow_spans(span, location));

                if use_in_later_iteration_of_loop {
                    let (later_use_kind, var_or_use_span, path_span) =
                        self.later_use_kind(borrow, spans, use_location);
                    BorrowExplanation::UsedLaterInLoop(later_use_kind, var_or_use_span, path_span)
                } else {
                    // Check if the location represents a `FakeRead`, and adapt the error
                    // message to the `FakeReadCause` it is from: in particular,
                    // the ones inserted in optimized `let var = <expr>` patterns.
                    let (later_use_kind, var_or_use_span, path_span) =
                        self.later_use_kind(borrow, spans, location);
                    BorrowExplanation::UsedLater(
                        borrow.borrowed_place.local,
                        later_use_kind,
                        var_or_use_span,
                        path_span,
                    )
                }
            }

            Some(Cause::DropVar(local, location)) => {
                let mut should_note_order = false;
                if self.local_name(local).is_some()
                    && let Some((WriteKind::StorageDeadOrDrop, place)) = kind_place
                    && let Some(borrowed_local) = place.as_local()
                    && self.local_name(borrowed_local).is_some()
                    && local != borrowed_local
                {
                    should_note_order = true;
                }

                BorrowExplanation::UsedLaterWhenDropped {
                    drop_loc: location,
                    dropped_local: local,
                    should_note_order,
                }
            }

            Some(Cause::LiveVar(..)) | None => {
                // Here, under NLL: no cause was found. Under polonius: no cause was found, or a
                // boring local was found, which we ignore like NLLs do to match its diagnostics.
                if let Some(region) = self.to_error_region_vid(borrow_region_vid) {
                    let (category, from_closure, span, region_name, path) =
                        self.free_region_constraint_info(borrow_region_vid, region);
                    if let Some(region_name) = region_name {
                        let opt_place_desc = self.describe_place(borrow.borrowed_place.as_ref());
                        BorrowExplanation::MustBeValidFor {
                            category,
                            from_closure,
                            span,
                            region_name,
                            opt_place_desc,
                            path,
                        }
                    } else {
                        debug!("Could not generate a region name");
                        BorrowExplanation::Unexplained
                    }
                } else {
                    debug!("Could not generate an error region vid");
                    BorrowExplanation::Unexplained
                }
            }
        }
    }

    /// Determine how the borrow was later used.
    /// First span returned points to the location of the conflicting use
    /// Second span if `Some` is returned in the case of closures and points
    /// to the use of the path
    #[instrument(level = "debug", skip(self))]
    fn later_use_kind(
        &self,
        borrow: &BorrowData<'tcx>,
        use_spans: UseSpans<'tcx>,
        location: Location,
    ) -> (LaterUseKind, Span, Option<Span>) {
        match use_spans {
            UseSpans::ClosureUse { capture_kind_span, path_span, .. } => {
                // Used in a closure.
                (LaterUseKind::ClosureCapture, capture_kind_span, Some(path_span))
            }
            // In the case that the borrowed value (probably a temporary)
            // overlaps with the method's receiver, then point at the method.
            UseSpans::FnSelfUse {
                var_span: span,
                kind: CallKind::Normal { desugaring: None, .. },
                ..
            } if span
                .overlaps(self.body.local_decls[borrow.assigned_place.local].source_info.span) =>
            {
                if let TerminatorKind::Call { func, call_source: CallSource::Normal, .. } =
                    &self.body.basic_blocks[location.block].terminator().kind
                {
                    // Just point to the function, to reduce the chance of overlapping spans.
                    let function_span = match func {
                        Operand::Constant(c) => c.span,
                        Operand::Copy(place) | Operand::Move(place) => {
                            if let Some(l) = place.as_local() {
                                let local_decl = &self.body.local_decls[l];
                                if self.local_name(l).is_none() {
                                    local_decl.source_info.span
                                } else {
                                    span
                                }
                            } else {
                                span
                            }
                        }
                    };
                    (LaterUseKind::Call, function_span, None)
                } else {
                    (LaterUseKind::Other, span, None)
                }
            }
            UseSpans::PatUse(span)
            | UseSpans::OtherUse(span)
            | UseSpans::FnSelfUse { var_span: span, .. } => {
                let block = &self.body.basic_blocks[location.block];

                let kind = if let Some(&Statement {
                    kind: StatementKind::FakeRead(box (FakeReadCause::ForLet(_), place)),
                    ..
                }) = block.statements.get(location.statement_index)
                {
                    if let Some(l) = place.as_local()
                        && let local_decl = &self.body.local_decls[l]
                        && local_decl.ty.is_closure()
                    {
                        LaterUseKind::ClosureCapture
                    } else {
                        LaterUseKind::FakeLetRead
                    }
                } else if self.was_captured_by_trait_object(borrow) {
                    LaterUseKind::TraitCapture
                } else if location.statement_index == block.statements.len() {
                    if let TerminatorKind::Call { func, call_source: CallSource::Normal, .. } =
                        &block.terminator().kind
                    {
                        // Just point to the function, to reduce the chance of overlapping spans.
                        let function_span = match func {
                            Operand::Constant(c) => c.span,
                            Operand::Copy(place) | Operand::Move(place) => {
                                if let Some(l) = place.as_local() {
                                    let local_decl = &self.body.local_decls[l];
                                    if self.local_name(l).is_none() {
                                        local_decl.source_info.span
                                    } else {
                                        span
                                    }
                                } else {
                                    span
                                }
                            }
                        };
                        return (LaterUseKind::Call, function_span, None);
                    } else {
                        LaterUseKind::Other
                    }
                } else {
                    LaterUseKind::Other
                };

                (kind, span, None)
            }
        }
    }

    /// Checks if a borrowed value was captured by a trait object. We do this by
    /// looking forward in the MIR from the reserve location and checking if we see
    /// an unsized cast to a trait object on our data.
    fn was_captured_by_trait_object(&self, borrow: &BorrowData<'tcx>) -> bool {
        // Start at the reserve location, find the place that we want to see cast to a trait object.
        let location = borrow.reserve_location;
        let block = &self.body[location.block];
        let stmt = block.statements.get(location.statement_index);
        debug!("was_captured_by_trait_object: location={:?} stmt={:?}", location, stmt);

        // We make a `queue` vector that has the locations we want to visit. As of writing, this
        // will only ever have one item at any given time, but by using a vector, we can pop from
        // it which simplifies the termination logic.
        let mut queue = vec![location];
        let mut target =
            if let Some(Statement { kind: StatementKind::Assign(box (place, _)), .. }) = stmt {
                if let Some(local) = place.as_local() {
                    local
                } else {
                    return false;
                }
            } else {
                return false;
            };

        debug!("was_captured_by_trait: target={:?} queue={:?}", target, queue);
        while let Some(current_location) = queue.pop() {
            debug!("was_captured_by_trait: target={:?}", target);
            let block = &self.body[current_location.block];
            // We need to check the current location to find out if it is a terminator.
            let is_terminator = current_location.statement_index == block.statements.len();
            if !is_terminator {
                let stmt = &block.statements[current_location.statement_index];
                debug!("was_captured_by_trait_object: stmt={:?}", stmt);

                // The only kind of statement that we care about is assignments...
                if let StatementKind::Assign(box (place, rvalue)) = &stmt.kind {
                    let Some(into) = place.local_or_deref_local() else {
                        // Continue at the next location.
                        queue.push(current_location.successor_within_block());
                        continue;
                    };

                    match rvalue {
                        // If we see a use, we should check whether it is our data, and if so
                        // update the place that we're looking for to that new place.
                        Rvalue::Use(operand) => match operand {
                            Operand::Copy(place) | Operand::Move(place) => {
                                if let Some(from) = place.as_local() {
                                    if from == target {
                                        target = into;
                                    }
                                }
                            }
                            _ => {}
                        },
                        // If we see an unsized cast, then if it is our data we should check
                        // whether it is being cast to a trait object.
                        Rvalue::Cast(
                            CastKind::PointerCoercion(PointerCoercion::Unsize, _),
                            operand,
                            ty,
                        ) => {
                            match operand {
                                Operand::Copy(place) | Operand::Move(place) => {
                                    if let Some(from) = place.as_local() {
                                        if from == target {
                                            debug!("was_captured_by_trait_object: ty={:?}", ty);
                                            // Check the type for a trait object.
                                            return match ty.kind() {
                                                // `&dyn Trait`
                                                ty::Ref(_, ty, _) if ty.is_trait() => true,
                                                // `Box<dyn Trait>`
                                                _ if ty.boxed_ty().is_some_and(Ty::is_trait) => {
                                                    true
                                                }

                                                // `dyn Trait`
                                                _ if ty.is_trait() => true,
                                                // Anything else.
                                                _ => false,
                                            };
                                        }
                                    }
                                    return false;
                                }
                                _ => return false,
                            }
                        }
                        _ => {}
                    }
                }

                // Continue at the next location.
                queue.push(current_location.successor_within_block());
            } else {
                // The only thing we need to do for terminators is progress to the next block.
                let terminator = block.terminator();
                debug!("was_captured_by_trait_object: terminator={:?}", terminator);

                if let TerminatorKind::Call { destination, target: Some(block), args, .. } =
                    &terminator.kind
                {
                    if let Some(dest) = destination.as_local() {
                        debug!(
                            "was_captured_by_trait_object: target={:?} dest={:?} args={:?}",
                            target, dest, args
                        );
                        // Check if one of the arguments to this function is the target place.
                        let found_target = args.iter().any(|arg| {
                            if let Operand::Move(place) = arg.node {
                                if let Some(potential) = place.as_local() {
                                    potential == target
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        });

                        // If it is, follow this to the next block and update the target.
                        if found_target {
                            target = dest;
                            queue.push(block.start_location());
                        }
                    }
                }
            }

            debug!("was_captured_by_trait: queue={:?}", queue);
        }

        // We didn't find anything and ran out of locations to check.
        false
    }
}
