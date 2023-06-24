use clippy_utils::{diagnostics::span_lint_and_help, is_from_proc_macro, path_to_local};
use rustc_hir::*;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::{lint::in_external_macro, ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// ### Why is this bad?
    ///
    /// ### Example
    /// ```rust,ignore
    /// let t1 = &[(1, 2), (3, 4)];
    /// let v1: Vec<[u32; 2]> = t1.iter().map(|&(a, b)| [a, b]).collect();
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// let t1 = &[(1, 2), (3, 4)];
    /// let v1: Vec<[u32; 2]> = t1.iter().map(|&t| t.into()).collect();
    /// ```
    #[clippy::version = "1.72.0"]
    pub TUPLE_ARRAY_CONVERSIONS,
    complexity,
    "default lint description"
}
declare_lint_pass!(TupleArrayConversions => [TUPLE_ARRAY_CONVERSIONS]);

impl LateLintPass<'_> for TupleArrayConversions {
    fn check_expr<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if !in_external_macro(cx.sess(), expr.span) {
            _ = check_array(cx, expr) || check_tuple(cx, expr);
        }
    }
}

fn check_array<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    let ExprKind::Array(elements) = expr.kind else {
        return false;
    };
    if !(1..=12).contains(&elements.len()) {
        return false;
    }

    if let Some(locals) = path_to_locals(cx, elements)
        && locals.iter().all(|local| {
            matches!(
                local,
                Node::Pat(pat) if matches!(
                    cx.typeck_results().pat_ty(backtrack_pat(cx, pat)).peel_refs().kind(),
                    ty::Tuple(_),
                ),
            )
        })
    {
        return emit_lint(cx, expr, ToType::Array);
    }

    if let Some(elements) = elements
            .iter()
            .map(|expr| {
                if let ExprKind::Field(path, _) = expr.kind {
                    return Some(path);
                };

                None
            })
            .collect::<Option<Vec<&Expr<'_>>>>()
        && let Some(locals) = path_to_locals(cx, elements)
        && locals.iter().all(|local| {
            matches!(
                local,
                Node::Pat(pat) if matches!(
                    cx.typeck_results().pat_ty(backtrack_pat(cx, pat)).peel_refs().kind(),
                    ty::Tuple(_),
                ),
            )
        })
    {
        return emit_lint(cx, expr, ToType::Array);
    }

    false
}

fn check_tuple<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    let ExprKind::Tup(elements) = expr.kind else {
        return false;
    };
    if !(1..=12).contains(&elements.len()) {
        return false;
    };
    if let Some(locals) = path_to_locals(cx, elements)
        && locals.iter().all(|local| {
            matches!(
                local,
                Node::Pat(pat) if matches!(
                    cx.typeck_results().pat_ty(backtrack_pat(cx, pat)).peel_refs().kind(),
                    ty::Array(_, _),
                ),
            )
        })
    {
        return emit_lint(cx, expr, ToType::Tuple);
    }

    if let Some(elements) = elements
            .iter()
            .map(|expr| {
                if let ExprKind::Index(path, _) = expr.kind {
                    return Some(path);
                };

                None
            })
            .collect::<Option<Vec<&Expr<'_>>>>()
        && let Some(locals) = path_to_locals(cx, elements.clone())
        && locals.iter().all(|local| {
            matches!(
                local,
                Node::Pat(pat) if cx.typeck_results()
                    .pat_ty(backtrack_pat(cx, pat))
                    .peel_refs()
                    .is_array()
            )
        })
    {
        return emit_lint(cx, expr, ToType::Tuple);
    }

    false
}

/// Walks up the `Pat` until it's reached the final containing `Pat`.
fn backtrack_pat<'tcx>(cx: &LateContext<'tcx>, start: &'tcx Pat<'tcx>) -> &'tcx Pat<'tcx> {
    let mut end = start;
    for (_, node) in cx.tcx.hir().parent_iter(start.hir_id) {
        if let Node::Pat(pat) = node {
            end = pat;
        } else {
            break;
        }
    }
    end
}

fn path_to_locals<'tcx>(
    cx: &LateContext<'tcx>,
    exprs: impl IntoIterator<Item = &'tcx Expr<'tcx>>,
) -> Option<Vec<Node<'tcx>>> {
    exprs
        .into_iter()
        .map(|element| path_to_local(element).and_then(|local| cx.tcx.hir().find(local)))
        .collect()
}

#[derive(Clone, Copy)]
enum ToType {
    Array,
    Tuple,
}

impl ToType {
    fn help(self) -> &'static str {
        match self {
            ToType::Array => "it looks like you're trying to convert a tuple to an array",
            ToType::Tuple => "it looks like you're trying to convert an array to a tuple",
        }
    }
}

fn emit_lint<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, to_type: ToType) -> bool {
    if !is_from_proc_macro(cx, expr) {
        span_lint_and_help(
            cx,
            TUPLE_ARRAY_CONVERSIONS,
            expr.span,
            to_type.help(),
            None,
            "use `.into()` instead",
        );

        return true;
    }

    false
}
