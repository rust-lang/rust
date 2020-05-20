use crate::utils;
use crate::utils::paths;
use crate::utils::sugg::Sugg;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Mutability, Param, Pat, PatKind, Path, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::Ident;

declare_clippy_lint! {
    /// **What it does:**
    /// Detects when people use `Vec::sort_by` and pass in a function
    /// which compares the second argument to the first.
    ///
    /// **Why is this bad?**
    /// It is more clear to use `Vec::sort_by_key` and `std::cmp::Reverse`
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// vec.sort_by(|a, b| b.foo().cmp(&a.foo()));
    /// ```
    /// Use instead:
    /// ```rust
    /// vec.sort_by_key(|e| Reverse(e.foo()));
    /// ```
    pub SORT_BY_KEY_REVERSE,
    complexity,
    "Use of `Vec::sort_by` when `Vec::sort_by_key` would be clearer"
}

declare_lint_pass!(SortByKeyReverse => [SORT_BY_KEY_REVERSE]);

struct LintTrigger {
    vec_name: String,
    closure_arg: String,
    closure_reverse_body: String,
    unstable: bool,
}

/// Detect if the two expressions are mirrored (identical, except one
/// contains a and the other replaces it with b)
fn mirrored_exprs(
    cx: &LateContext<'_, '_>,
    a_expr: &Expr<'_>,
    a_ident: &Ident,
    b_expr: &Expr<'_>,
    b_ident: &Ident,
) -> bool {
    match (&a_expr.kind, &b_expr.kind) {
        // Two boxes with mirrored contents
        (ExprKind::Box(left_expr), ExprKind::Box(right_expr)) => {
            mirrored_exprs(cx, left_expr, a_ident, right_expr, b_ident)
        },
        // Two arrays with mirrored contents
        (ExprKind::Array(left_exprs), ExprKind::Array(right_exprs)) => left_exprs
            .iter()
            .zip(right_exprs.iter())
            .all(|(left, right)| mirrored_exprs(cx, left, a_ident, right, b_ident)),
        // The two exprs are function calls.
        // Check to see that the function itself and its arguments are mirrored
        (ExprKind::Call(left_expr, left_args), ExprKind::Call(right_expr, right_args)) => {
            mirrored_exprs(cx, left_expr, a_ident, right_expr, b_ident)
                && left_args
                    .iter()
                    .zip(right_args.iter())
                    .all(|(left, right)| mirrored_exprs(cx, left, a_ident, right, b_ident))
        },
        // The two exprs are method calls.
        // Check to see that the function is the same and the arguments are mirrored
        // This is enough because the receiver of the method is listed in the arguments
        (ExprKind::MethodCall(left_segment, _, left_args), ExprKind::MethodCall(right_segment, _, right_args)) => {
            left_segment.ident == right_segment.ident
                && left_args
                    .iter()
                    .zip(right_args.iter())
                    .all(|(left, right)| mirrored_exprs(cx, left, a_ident, right, b_ident))
        },
        // Two tuples with mirrored contents
        (ExprKind::Tup(left_exprs), ExprKind::Tup(right_exprs)) => left_exprs
            .iter()
            .zip(right_exprs.iter())
            .all(|(left, right)| mirrored_exprs(cx, left, a_ident, right, b_ident)),
        // Two binary ops, which are the same operation and which have mirrored arguments
        (ExprKind::Binary(left_op, left_left, left_right), ExprKind::Binary(right_op, right_left, right_right)) => {
            left_op.node == right_op.node
                && mirrored_exprs(cx, left_left, a_ident, right_left, b_ident)
                && mirrored_exprs(cx, left_right, a_ident, right_right, b_ident)
        },
        // Two unary ops, which are the same operation and which have the same argument
        (ExprKind::Unary(left_op, left_expr), ExprKind::Unary(right_op, right_expr)) => {
            left_op == right_op && mirrored_exprs(cx, left_expr, a_ident, right_expr, b_ident)
        },
        // The two exprs are literals of some kind
        (ExprKind::Lit(left_lit), ExprKind::Lit(right_lit)) => left_lit.node == right_lit.node,
        (ExprKind::Cast(left, _), ExprKind::Cast(right, _)) => mirrored_exprs(cx, left, a_ident, right, b_ident),
        (ExprKind::DropTemps(left_block), ExprKind::DropTemps(right_block)) => {
            mirrored_exprs(cx, left_block, a_ident, right_block, b_ident)
        },
        (ExprKind::Field(left_expr, left_ident), ExprKind::Field(right_expr, right_ident)) => {
            left_ident.name == right_ident.name && mirrored_exprs(cx, left_expr, a_ident, right_expr, right_ident)
        },
        // Two paths: either one is a and the other is b, or they're identical to each other
        (
            ExprKind::Path(QPath::Resolved(
                _,
                Path {
                    segments: left_segments,
                    ..
                },
            )),
            ExprKind::Path(QPath::Resolved(
                _,
                Path {
                    segments: right_segments,
                    ..
                },
            )),
        ) => {
            (left_segments
                .iter()
                .zip(right_segments.iter())
                .all(|(left, right)| left.ident == right.ident)
                && left_segments
                    .iter()
                    .all(|seg| &seg.ident != a_ident && &seg.ident != b_ident))
                || (left_segments.len() == 1
                    && &left_segments[0].ident == a_ident
                    && right_segments.len() == 1
                    && &right_segments[0].ident == b_ident)
        },
        // Matching expressions, but one or both is borrowed
        (
            ExprKind::AddrOf(left_kind, Mutability::Not, left_expr),
            ExprKind::AddrOf(right_kind, Mutability::Not, right_expr),
        ) => left_kind == right_kind && mirrored_exprs(cx, left_expr, a_ident, right_expr, b_ident),
        (_, ExprKind::AddrOf(_, Mutability::Not, right_expr)) => {
            mirrored_exprs(cx, a_expr, a_ident, right_expr, b_ident)
        },
        (ExprKind::AddrOf(_, Mutability::Not, left_expr), _) => mirrored_exprs(cx, left_expr, a_ident, b_expr, b_ident),
        _ => false,
    }
}

fn detect_lint(cx: &LateContext<'_, '_>, expr: &Expr<'_>) -> Option<LintTrigger> {
    if_chain! {
        if let ExprKind::MethodCall(name_ident, _, args) = &expr.kind;
        if let name = name_ident.ident.name.to_ident_string();
        if name == "sort_by" || name == "sort_unstable_by";
        if let [vec, Expr { kind: ExprKind::Closure(_, _, closure_body_id, _, _), .. }] = args;
        if utils::match_type(cx, &cx.tables.expr_ty(vec), &paths::VEC);
        if let closure_body = cx.tcx.hir().body(*closure_body_id);
        if let &[
            Param { pat: Pat { kind: PatKind::Binding(_, _, a_ident, _), .. }, ..},
            Param { pat: Pat { kind: PatKind::Binding(_, _, b_ident, _), .. }, .. }
        ] = &closure_body.params;
        if let ExprKind::MethodCall(method_path, _, [ref b_expr, ref a_expr]) = &closure_body.value.kind;
        if method_path.ident.name.to_ident_string() == "cmp";
        if mirrored_exprs(&cx, &a_expr, &a_ident, &b_expr, &b_ident);
        then {
            let vec_name = Sugg::hir(cx, &args[0], "..").to_string();
            let unstable = name == "sort_unstable_by";
            let closure_arg = format!("&{}", b_ident.name.to_ident_string());
            let closure_reverse_body = Sugg::hir(cx, &b_expr, "..").to_string();
            // Get rid of parentheses, because they aren't needed anymore
            // while closure_reverse_body.chars().next() == Some('(') && closure_reverse_body.chars().last() == Some(')') {
                // closure_reverse_body = String::from(&closure_reverse_body[1..closure_reverse_body.len()-1]);
            // }
            Some(LintTrigger { vec_name, unstable, closure_arg, closure_reverse_body })
        } else {
            None
        }
    }
}

impl LateLintPass<'_, '_> for SortByKeyReverse {
    fn check_expr(&mut self, cx: &LateContext<'_, '_>, expr: &Expr<'_>) {
        if let Some(trigger) = detect_lint(cx, expr) {
            utils::span_lint_and_sugg(
                cx,
                SORT_BY_KEY_REVERSE,
                expr.span,
                "use Vec::sort_by_key here instead",
                "try",
                format!(
                    "{}.sort{}_by_key(|{}| Reverse({}))",
                    trigger.vec_name,
                    if trigger.unstable { "_unstable" } else { "" },
                    trigger.closure_arg,
                    trigger.closure_reverse_body,
                ),
                Applicability::MachineApplicable,
            );
        }
    }
}
