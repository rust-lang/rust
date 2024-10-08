use super::{EXPLICIT_COUNTER_LOOP, IncrementVisitor, InitializeVisitor, make_iterator_snippet};
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{get_enclosing_block, is_integer_const};
use rustc_ast::Label;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_block, walk_expr};
use rustc_hir::{Expr, Pat};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, UintTy};

// To trigger the EXPLICIT_COUNTER_LOOP lint, a variable must be
// incremented exactly once in the loop body, and initialized to zero
// at the start of the loop.
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
    if let Some(block) = get_enclosing_block(cx, expr.hir_id) {
        for id in increment_visitor.into_results() {
            let mut initialize_visitor = InitializeVisitor::new(cx, expr, id);
            walk_block(&mut initialize_visitor, block);

            if let Some((name, ty, initializer)) = initialize_visitor.get_result()
                && is_integer_const(cx, initializer, 0)
            {
                let mut applicability = Applicability::MaybeIncorrect;
                let span = expr.span.with_hi(arg.span.hi());
                let loop_label = label.map_or(String::new(), |l| format!("{}: ", l.ident.name));
                let int_name = match ty.map(Ty::kind) {
                    // usize or inferred
                    Some(ty::Uint(UintTy::Usize)) | None => {
                        span_lint_and_sugg(
                            cx,
                            EXPLICIT_COUNTER_LOOP,
                            span,
                            format!("the variable `{name}` is used as a loop counter"),
                            "consider using",
                            format!(
                                "{loop_label}for ({name}, {}) in {}.enumerate()",
                                snippet_with_applicability(cx, pat.span, "item", &mut applicability),
                                make_iterator_snippet(cx, arg, &mut applicability),
                            ),
                            applicability,
                        );
                        return;
                    },
                    Some(ty::Int(int_ty)) => int_ty.name_str(),
                    Some(ty::Uint(uint_ty)) => uint_ty.name_str(),
                    _ => return,
                };

                span_lint_and_then(
                    cx,
                    EXPLICIT_COUNTER_LOOP,
                    span,
                    format!("the variable `{name}` is used as a loop counter"),
                    |diag| {
                        diag.span_suggestion(
                            span,
                            "consider using",
                            format!(
                                "{loop_label}for ({name}, {}) in (0_{int_name}..).zip({})",
                                snippet_with_applicability(cx, pat.span, "item", &mut applicability),
                                make_iterator_snippet(cx, arg, &mut applicability),
                            ),
                            applicability,
                        );

                        diag.note(format!(
                            "`{name}` is of type `{int_name}`, making it ineligible for `Iterator::enumerate`"
                        ));
                    },
                );
            }
        }
    }
}
