use std::borrow::Cow;

use super::{EXPLICIT_COUNTER_LOOP, IncrementVisitor, InitializeVisitor, make_iterator_snippet};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::{EMPTY, Sugg};
use clippy_utils::{get_enclosing_block, is_integer_const, is_integer_literal_untyped};
use rustc_ast::{Label, RangeLimits};
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_block, walk_expr};
use rustc_hir::{Expr, Pat};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, UintTy};

// To trigger the EXPLICIT_COUNTER_LOOP lint, a variable must be incremented exactly once in the
// loop body.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    arg: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    expr: &'tcx Expr<'_>,
    label: Option<Label>,
) {
    // Look for variables that are incremented once per loop iteration.
    let mut increment_visitor = IncrementVisitor::new(cx);
    walk_expr(&mut increment_visitor, body);

    // For each candidate, check the parent block to see if
    // it's initialized to zero at the start of the loop.
    let Some(block) = get_enclosing_block(cx, expr.hir_id) else {
        return;
    };

    for id in increment_visitor.into_results() {
        let mut initialize_visitor = InitializeVisitor::new(cx, expr, id);
        walk_block(&mut initialize_visitor, block);

        let Some((name, ty, initializer)) = initialize_visitor.get_result() else {
            continue;
        };
        if !cx.typeck_results().expr_ty(initializer).is_integral() {
            continue;
        }

        let is_zero = is_integer_const(cx, initializer, 0);
        let mut applicability = Applicability::MaybeIncorrect;
        let span = expr.span.with_hi(arg.span.hi());
        let loop_label = label.map_or(String::new(), |l| format!("{}: ", l.ident.name));

        span_lint_and_then(
            cx,
            EXPLICIT_COUNTER_LOOP,
            span,
            format!("the variable `{name}` is used as a loop counter"),
            |diag| {
                let pat_snippet = snippet_with_applicability(cx, pat.span, "item", &mut applicability);
                let iter_snippet = make_iterator_snippet(cx, arg, &mut applicability);
                let int_name = match ty.map(Ty::kind) {
                    Some(ty::Uint(UintTy::Usize)) | None => {
                        if is_zero {
                            diag.span_suggestion(
                                span,
                                "consider using",
                                format!("{loop_label}for ({name}, {pat_snippet}) in {iter_snippet}.enumerate()"),
                                applicability,
                            );
                            return;
                        }
                        None
                    },
                    Some(ty::Int(int_ty)) => Some(int_ty.name_str()),
                    Some(ty::Uint(uint_ty)) => Some(uint_ty.name_str()),
                    _ => None,
                }
                .filter(|_| is_integer_literal_untyped(initializer));

                let initializer = Sugg::hir_from_snippet(cx, initializer, |span| {
                    let snippet = snippet_with_applicability(cx, span, "..", &mut applicability);
                    if let Some(int_name) = int_name {
                        return Cow::Owned(format!("{snippet}_{int_name}"));
                    }
                    snippet
                });

                diag.span_suggestion(
                    span,
                    "consider using",
                    format!(
                        "{loop_label}for ({name}, {pat_snippet}) in ({}).zip({iter_snippet})",
                        initializer.range(&EMPTY, RangeLimits::HalfOpen)
                    ),
                    applicability,
                );

                if is_zero && let Some(int_name) = int_name {
                    diag.note(format!(
                        "`{name}` is of type `{int_name}`, making it ineligible for `Iterator::enumerate`"
                    ));
                }
            },
        );
    }
}
