use crate::{LateContext, LateLintPass, LintContext};

use hir::{Expr, Pat};
use rustc_errors::{Applicability, DelayDm};
use rustc_hir as hir;
use rustc_infer::traits::TraitEngine;
use rustc_infer::{infer::TyCtxtInferExt, traits::ObligationCause};
use rustc_middle::ty::{self, List};
use rustc_span::{sym, Span};
use rustc_trait_selection::traits::TraitEngineExt;

declare_lint! {
    /// The `for_loops_over_fallibles` lint checks for `for` loops over `Option` or `Result` values.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let opt = Some(1);
    /// for x in opt { /* ... */}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Both `Option` and `Result` implement `IntoIterator` trait, which allows using them in a `for` loop.
    /// `for` loop over `Option` or `Result` will iterate either 0 (if the value is `None`/`Err(_)`)
    /// or 1 time (if the value is `Some(_)`/`Ok(_)`). This is not very useful and is more clearly expressed
    /// via `if let`.
    ///
    /// `for` loop can also be accidentally written with the intention to call a function multiple times,
    /// while the function returns `Some(_)`, in these cases `while let` loop should be used instead.
    ///
    /// The "intended" use of `IntoIterator` implementations for `Option` and `Result` is passing them to
    /// generic code that expects something implementing `IntoIterator`. For example using `.chain(option)`
    /// to optionally add a value to an iterator.
    pub FOR_LOOPS_OVER_FALLIBLES,
    Warn,
    "for-looping over an `Option` or a `Result`, which is more clearly expressed as an `if let`"
}

declare_lint_pass!(ForLoopsOverFallibles => [FOR_LOOPS_OVER_FALLIBLES]);

impl<'tcx> LateLintPass<'tcx> for ForLoopsOverFallibles {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let Some((pat, arg)) = extract_for_loop(expr) else { return };

        let ty = cx.typeck_results().expr_ty(arg);

        let &ty::Adt(adt, substs) = ty.kind() else { return };

        let (article, ty, var) = match adt.did() {
            did if cx.tcx.is_diagnostic_item(sym::Option, did) => ("an", "Option", "Some"),
            did if cx.tcx.is_diagnostic_item(sym::Result, did) => ("a", "Result", "Ok"),
            _ => return,
        };

        let msg = DelayDm(|| {
            format!(
                "for loop over {article} `{ty}`. This is more readably written as an `if let` statement",
            )
        });

        cx.struct_span_lint(FOR_LOOPS_OVER_FALLIBLES, arg.span, msg, |lint| {
            if let Some(recv) = extract_iterator_next_call(cx, arg)
            && let Ok(recv_snip) = cx.sess().source_map().span_to_snippet(recv.span)
            {
                lint.span_suggestion(
                    recv.span.between(arg.span.shrink_to_hi()),
                    format!("to iterate over `{recv_snip}` remove the call to `next`"),
                    ".by_ref()",
                    Applicability::MaybeIncorrect
                );
            } else {
                lint.multipart_suggestion_verbose(
                    format!("to check pattern in a loop use `while let`"),
                    vec![
                        // NB can't use `until` here because `expr.span` and `pat.span` have different syntax contexts
                        (expr.span.with_hi(pat.span.lo()), format!("while let {var}(")),
                        (pat.span.between(arg.span), format!(") = ")),
                    ],
                    Applicability::MaybeIncorrect
                );
            }

            if suggest_question_mark(cx, adt, substs, expr.span) {
                lint.span_suggestion(
                    arg.span.shrink_to_hi(),
                    "consider unwrapping the `Result` with `?` to iterate over its contents",
                    "?",
                    Applicability::MaybeIncorrect,
                );
            }

            lint.multipart_suggestion_verbose(
                "consider using `if let` to clear intent",
                vec![
                    // NB can't use `until` here because `expr.span` and `pat.span` have different syntax contexts
                    (expr.span.with_hi(pat.span.lo()), format!("if let {var}(")),
                    (pat.span.between(arg.span), format!(") = ")),
                ],
                Applicability::MaybeIncorrect,
            )
        })
    }
}

fn extract_for_loop<'tcx>(expr: &Expr<'tcx>) -> Option<(&'tcx Pat<'tcx>, &'tcx Expr<'tcx>)> {
    if let hir::ExprKind::DropTemps(e) = expr.kind
    && let hir::ExprKind::Match(iterexpr, [arm], hir::MatchSource::ForLoopDesugar) = e.kind
    && let hir::ExprKind::Call(_, [arg]) = iterexpr.kind
    && let hir::ExprKind::Loop(block, ..) = arm.body.kind
    && let [stmt] = block.stmts
    && let hir::StmtKind::Expr(e) = stmt.kind
    && let hir::ExprKind::Match(_, [_, some_arm], _) = e.kind
    && let hir::PatKind::Struct(_, [field], _) = some_arm.pat.kind
    {
        Some((field.pat, arg))
    } else {
        None
    }
}

fn extract_iterator_next_call<'tcx>(
    cx: &LateContext<'_>,
    expr: &Expr<'tcx>,
) -> Option<&'tcx Expr<'tcx>> {
    // This won't work for `Iterator::next(iter)`, is this an issue?
    if let hir::ExprKind::MethodCall(_, recv, _, _) = expr.kind
    && cx.typeck_results().type_dependent_def_id(expr.hir_id) == cx.tcx.lang_items().next_fn()
    {
        Some(recv)
    } else {
        return None
    }
}

fn suggest_question_mark<'tcx>(
    cx: &LateContext<'tcx>,
    adt: ty::AdtDef<'tcx>,
    substs: &List<ty::GenericArg<'tcx>>,
    span: Span,
) -> bool {
    let Some(body_id) = cx.enclosing_body else { return false };
    let Some(into_iterator_did) = cx.tcx.get_diagnostic_item(sym::IntoIterator) else { return false };

    if !cx.tcx.is_diagnostic_item(sym::Result, adt.did()) {
        return false;
    }

    // Check that the function/closure/constant we are in has a `Result` type.
    // Otherwise suggesting using `?` may not be a good idea.
    {
        let ty = cx.typeck_results().expr_ty(&cx.tcx.hir().body(body_id).value);
        let ty::Adt(ret_adt, ..) = ty.kind() else { return false };
        if !cx.tcx.is_diagnostic_item(sym::Result, ret_adt.did()) {
            return false;
        }
    }

    let ty = substs.type_at(0);
    let infcx = cx.tcx.infer_ctxt().build();
    let mut fulfill_cx = <dyn TraitEngine<'_>>::new(infcx.tcx);

    let cause = ObligationCause::new(
        span,
        body_id.hir_id,
        rustc_infer::traits::ObligationCauseCode::MiscObligation,
    );
    fulfill_cx.register_bound(
        &infcx,
        ty::ParamEnv::empty(),
        // Erase any region vids from the type, which may not be resolved
        infcx.tcx.erase_regions(ty),
        into_iterator_did,
        cause,
    );

    // Select all, including ambiguous predicates
    let errors = fulfill_cx.select_all_or_error(&infcx);

    errors.is_empty()
}
