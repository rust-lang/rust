use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Mutability, Param, Pat, PatKind, Path, PathSegment, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, subst::GenericArgKind};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;
use rustc_span::symbol::Ident;
use std::iter;

declare_clippy_lint! {
    /// ### What it does
    /// Detects uses of `Vec::sort_by` passing in a closure
    /// which compares the two arguments, either directly or indirectly.
    ///
    /// ### Why is this bad?
    /// It is more clear to use `Vec::sort_by_key` (or `Vec::sort` if
    /// possible) than to use `Vec::sort_by` and a more complicated
    /// closure.
    ///
    /// ### Known problems
    /// If the suggested `Vec::sort_by_key` uses Reverse and it isn't already
    /// imported by a use statement, then it will need to be added manually.
    ///
    /// ### Example
    /// ```rust
    /// # struct A;
    /// # impl A { fn foo(&self) {} }
    /// # let mut vec: Vec<A> = Vec::new();
    /// vec.sort_by(|a, b| a.foo().cmp(&b.foo()));
    /// ```
    /// Use instead:
    /// ```rust
    /// # struct A;
    /// # impl A { fn foo(&self) {} }
    /// # let mut vec: Vec<A> = Vec::new();
    /// vec.sort_by_key(|a| a.foo());
    /// ```
    #[clippy::version = "1.46.0"]
    pub UNNECESSARY_SORT_BY,
    complexity,
    "Use of `Vec::sort_by` when `Vec::sort_by_key` or `Vec::sort` would be clearer"
}

declare_lint_pass!(UnnecessarySortBy => [UNNECESSARY_SORT_BY]);

enum LintTrigger {
    Sort(SortDetection),
    SortByKey(SortByKeyDetection),
}

struct SortDetection {
    vec_name: String,
    unstable: bool,
}

struct SortByKeyDetection {
    vec_name: String,
    closure_arg: String,
    closure_body: String,
    reverse: bool,
    unstable: bool,
}

/// Detect if the two expressions are mirrored (identical, except one
/// contains a and the other replaces it with b)
fn mirrored_exprs(
    cx: &LateContext<'_>,
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
        (ExprKind::Array(left_exprs), ExprKind::Array(right_exprs)) => {
            iter::zip(*left_exprs, *right_exprs).all(|(left, right)| mirrored_exprs(cx, left, a_ident, right, b_ident))
        },
        // The two exprs are function calls.
        // Check to see that the function itself and its arguments are mirrored
        (ExprKind::Call(left_expr, left_args), ExprKind::Call(right_expr, right_args)) => {
            mirrored_exprs(cx, left_expr, a_ident, right_expr, b_ident)
                && iter::zip(*left_args, *right_args)
                    .all(|(left, right)| mirrored_exprs(cx, left, a_ident, right, b_ident))
        },
        // The two exprs are method calls.
        // Check to see that the function is the same and the arguments are mirrored
        // This is enough because the receiver of the method is listed in the arguments
        (ExprKind::MethodCall(left_segment, left_args, _), ExprKind::MethodCall(right_segment, right_args, _)) => {
            left_segment.ident == right_segment.ident
                && iter::zip(*left_args, *right_args)
                    .all(|(left, right)| mirrored_exprs(cx, left, a_ident, right, b_ident))
        },
        // Two tuples with mirrored contents
        (ExprKind::Tup(left_exprs), ExprKind::Tup(right_exprs)) => {
            iter::zip(*left_exprs, *right_exprs).all(|(left, right)| mirrored_exprs(cx, left, a_ident, right, b_ident))
        },
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
            (iter::zip(*left_segments, *right_segments).all(|(left, right)| left.ident == right.ident)
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

fn detect_lint(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<LintTrigger> {
    if_chain! {
        if let ExprKind::MethodCall(name_ident, args, _) = &expr.kind;
        if let name = name_ident.ident.name.to_ident_string();
        if name == "sort_by" || name == "sort_unstable_by";
        if let [vec, Expr { kind: ExprKind::Closure(_, _, closure_body_id, _, _), .. }] = args;
        if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(vec), sym::Vec);
        if let closure_body = cx.tcx.hir().body(*closure_body_id);
        if let &[
            Param { pat: Pat { kind: PatKind::Binding(_, _, left_ident, _), .. }, ..},
            Param { pat: Pat { kind: PatKind::Binding(_, _, right_ident, _), .. }, .. }
        ] = &closure_body.params;
        if let ExprKind::MethodCall(method_path, [ref left_expr, ref right_expr], _) = &closure_body.value.kind;
        if method_path.ident.name == sym::cmp;
        then {
            let (closure_body, closure_arg, reverse) = if mirrored_exprs(
                cx,
                left_expr,
                left_ident,
                right_expr,
                right_ident
            ) {
                (Sugg::hir(cx, left_expr, "..").to_string(), left_ident.name.to_string(), false)
            } else if mirrored_exprs(cx, left_expr, right_ident, right_expr, left_ident) {
                (Sugg::hir(cx, left_expr, "..").to_string(), right_ident.name.to_string(), true)
            } else {
                return None;
            };
            let vec_name = Sugg::hir(cx, &args[0], "..").to_string();
            let unstable = name == "sort_unstable_by";

            if_chain! {
            if let ExprKind::Path(QPath::Resolved(_, Path {
                segments: [PathSegment { ident: left_name, .. }], ..
            })) = &left_expr.kind;
            if left_name == left_ident;
            if cx.tcx.get_diagnostic_item(sym::Ord).map_or(false, |id| {
                implements_trait(cx, cx.typeck_results().expr_ty(left_expr), id, &[])
            });
                then {
                    return Some(LintTrigger::Sort(SortDetection { vec_name, unstable }));
                }
            }

            if !expr_borrows(cx, left_expr) {
                return Some(LintTrigger::SortByKey(SortByKeyDetection {
                    vec_name,
                    closure_arg,
                    closure_body,
                    reverse,
                    unstable,
                }));
            }
        }
    }

    None
}

fn expr_borrows(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let ty = cx.typeck_results().expr_ty(expr);
    matches!(ty.kind(), ty::Ref(..)) || ty.walk().any(|arg| matches!(arg.unpack(), GenericArgKind::Lifetime(_)))
}

impl LateLintPass<'_> for UnnecessarySortBy {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        match detect_lint(cx, expr) {
            Some(LintTrigger::SortByKey(trigger)) => span_lint_and_sugg(
                cx,
                UNNECESSARY_SORT_BY,
                expr.span,
                "use Vec::sort_by_key here instead",
                "try",
                format!(
                    "{}.sort{}_by_key(|{}| {})",
                    trigger.vec_name,
                    if trigger.unstable { "_unstable" } else { "" },
                    trigger.closure_arg,
                    if trigger.reverse {
                        format!("Reverse({})", trigger.closure_body)
                    } else {
                        trigger.closure_body.to_string()
                    },
                ),
                if trigger.reverse {
                    Applicability::MaybeIncorrect
                } else {
                    Applicability::MachineApplicable
                },
            ),
            Some(LintTrigger::Sort(trigger)) => span_lint_and_sugg(
                cx,
                UNNECESSARY_SORT_BY,
                expr.span,
                "use Vec::sort here instead",
                "try",
                format!(
                    "{}.sort{}()",
                    trigger.vec_name,
                    if trigger.unstable { "_unstable" } else { "" },
                ),
                Applicability::MachineApplicable,
            ),
            None => {},
        }
    }
}
