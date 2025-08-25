use hir::{Expr, Pat};
use rustc_hir::{self as hir, LangItem};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::ObligationCause;
use rustc_middle::ty;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::{Span, sym};
use rustc_trait_selection::traits::ObligationCtxt;

use crate::lints::{
    ForLoopsOverFalliblesDiag, ForLoopsOverFalliblesLoopSub, ForLoopsOverFalliblesQuestionMark,
    ForLoopsOverFalliblesSuggestion,
};
use crate::{LateContext, LateLintPass, LintContext};

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

        let arg_span = arg.span.source_callsite();

        let ty = cx.typeck_results().expr_ty(arg);

        let (adt, args, ref_mutability) = match ty.kind() {
            &ty::Adt(adt, args) => (adt, args, None),
            &ty::Ref(_, ty, mutability) => match ty.kind() {
                &ty::Adt(adt, args) => (adt, args, Some(mutability)),
                _ => return,
            },
            _ => return,
        };

        let (article, ty, var) = match adt.did() {
            did if cx.tcx.is_diagnostic_item(sym::Option, did) && ref_mutability.is_some() => {
                ("a", "Option", "Some")
            }
            did if cx.tcx.is_diagnostic_item(sym::Option, did) => ("an", "Option", "Some"),
            did if cx.tcx.is_diagnostic_item(sym::Result, did) => ("a", "Result", "Ok"),
            _ => return,
        };

        let ref_prefix = match ref_mutability {
            None => "",
            Some(ref_mutability) => ref_mutability.ref_prefix_str(),
        };

        let sub = if let Some(recv) = extract_iterator_next_call(cx, arg)
            && let Ok(recv_snip) = cx.sess().source_map().span_to_snippet(recv.span)
        {
            ForLoopsOverFalliblesLoopSub::RemoveNext {
                suggestion: recv.span.between(arg_span.shrink_to_hi()),
                recv_snip,
            }
        } else {
            ForLoopsOverFalliblesLoopSub::UseWhileLet {
                start_span: expr.span.with_hi(pat.span.lo()),
                end_span: pat.span.between(arg_span),
                var,
            }
        };
        let question_mark = suggest_question_mark(cx, adt, args, expr.span)
            .then(|| ForLoopsOverFalliblesQuestionMark { suggestion: arg_span.shrink_to_hi() });
        let suggestion = ForLoopsOverFalliblesSuggestion {
            var,
            start_span: expr.span.with_hi(pat.span.lo()),
            end_span: pat.span.between(arg_span),
        };

        cx.emit_span_lint(
            FOR_LOOPS_OVER_FALLIBLES,
            arg_span,
            ForLoopsOverFalliblesDiag { article, ref_prefix, ty, sub, question_mark, suggestion },
        );
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
        && cx
            .typeck_results()
            .type_dependent_def_id(expr.hir_id)
            .is_some_and(|def_id| cx.tcx.is_lang_item(def_id, LangItem::IteratorNext))
    {
        Some(recv)
    } else {
        None
    }
}

fn suggest_question_mark<'tcx>(
    cx: &LateContext<'tcx>,
    adt: ty::AdtDef<'tcx>,
    args: ty::GenericArgsRef<'tcx>,
    span: Span,
) -> bool {
    let Some(body_id) = cx.enclosing_body else { return false };
    let Some(into_iterator_did) = cx.tcx.get_diagnostic_item(sym::IntoIterator) else {
        return false;
    };

    if !cx.tcx.is_diagnostic_item(sym::Result, adt.did()) {
        return false;
    }

    // Check that the function/closure/constant we are in has a `Result` type.
    // Otherwise suggesting using `?` may not be a good idea.
    {
        let ty = cx.typeck_results().expr_ty(cx.tcx.hir_body(body_id).value);
        let ty::Adt(ret_adt, ..) = ty.kind() else { return false };
        if !cx.tcx.is_diagnostic_item(sym::Result, ret_adt.did()) {
            return false;
        }
    }

    let ty = args.type_at(0);
    let (infcx, param_env) = cx.tcx.infer_ctxt().build_with_typing_env(cx.typing_env());
    let ocx = ObligationCtxt::new(&infcx);

    let body_def_id = cx.tcx.hir_body_owner_def_id(body_id);
    let cause =
        ObligationCause::new(span, body_def_id, rustc_infer::traits::ObligationCauseCode::Misc);

    ocx.register_bound(
        cause,
        param_env,
        // Erase any region vids from the type, which may not be resolved
        infcx.tcx.erase_and_anonymize_regions(ty),
        into_iterator_did,
    );

    ocx.select_all_or_error().is_empty()
}
