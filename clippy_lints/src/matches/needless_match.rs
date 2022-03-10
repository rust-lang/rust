use super::NEEDLESS_MATCH;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{eq_expr_value, get_parent_expr, higher, is_else_clause, is_lang_ctor, peel_blocks_with_stmt};
use rustc_errors::Applicability;
use rustc_hir::LangItem::OptionNone;
use rustc_hir::{Arm, BindingAnnotation, Expr, ExprKind, Pat, PatKind, Path, PathSegment, QPath, UnOp};
use rustc_lint::LateContext;
use rustc_span::sym;

pub(crate) fn check_match(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>]) {
    // This is for avoiding collision with `match_single_binding`.
    if arms.len() < 2 {
        return;
    }

    for arm in arms {
        if let PatKind::Wild = arm.pat.kind {
            let ret_expr = strip_return(arm.body);
            if !eq_expr_value(cx, ex, ret_expr) {
                return;
            }
        } else if !pat_same_as_expr(arm.pat, arm.body) {
            return;
        }
    }

    if let Some(match_expr) = get_parent_expr(cx, ex) {
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            NEEDLESS_MATCH,
            match_expr.span,
            "this match expression is unnecessary",
            "replace it with",
            snippet_with_applicability(cx, ex.span, "..", &mut applicability).to_string(),
            applicability,
        );
    }
}

/// Check for nop `if let` expression that assembled as unnecessary match
///
/// ```rust,ignore
/// if let Some(a) = option {
///     Some(a)
/// } else {
///     None
/// }
/// ```
/// OR
/// ```rust,ignore
/// if let SomeEnum::A = some_enum {
///     SomeEnum::A
/// } else if let SomeEnum::B = some_enum {
///     SomeEnum::B
/// } else {
///     some_enum
/// }
/// ```
pub(crate) fn check(cx: &LateContext<'_>, ex: &Expr<'_>) {
    if_chain! {
        if let Some(ref if_let) = higher::IfLet::hir(cx, ex);
        if !is_else_clause(cx.tcx, ex);
        if check_if_let(cx, if_let);
        then {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                NEEDLESS_MATCH,
                ex.span,
                "this if-let expression is unnecessary",
                "replace it with",
                snippet_with_applicability(cx, if_let.let_expr.span, "..", &mut applicability).to_string(),
                applicability,
            );
        }
    }
}

fn check_if_let(cx: &LateContext<'_>, if_let: &higher::IfLet<'_>) -> bool {
    if let Some(if_else) = if_let.if_else {
        if !pat_same_as_expr(if_let.let_pat, peel_blocks_with_stmt(if_let.if_then)) {
            return false;
        }

        // Recurrsively check for each `else if let` phrase,
        if let Some(ref nested_if_let) = higher::IfLet::hir(cx, if_else) {
            return check_if_let(cx, nested_if_let);
        }

        if matches!(if_else.kind, ExprKind::Block(..)) {
            let else_expr = peel_blocks_with_stmt(if_else);
            let ret = strip_return(else_expr);
            let let_expr_ty = cx.typeck_results().expr_ty(if_let.let_expr);
            if is_type_diagnostic_item(cx, let_expr_ty, sym::Option) {
                if let ExprKind::Path(ref qpath) = ret.kind {
                    return is_lang_ctor(cx, qpath, OptionNone) || eq_expr_value(cx, if_let.let_expr, ret);
                }
            } else {
                return eq_expr_value(cx, if_let.let_expr, ret);
            }
            return true;
        }
    }
    false
}

/// Strip `return` keyword if the expression type is `ExprKind::Ret`.
fn strip_return<'hir>(expr: &'hir Expr<'hir>) -> &'hir Expr<'hir> {
    if let ExprKind::Ret(Some(ret)) = expr.kind {
        ret
    } else {
        expr
    }
}

fn pat_same_as_expr(pat: &Pat<'_>, expr: &Expr<'_>) -> bool {
    let expr = strip_return(expr);
    match (&pat.kind, &expr.kind) {
        // Example: `Some(val) => Some(val)`
        (
            PatKind::TupleStruct(QPath::Resolved(_, path), [first_pat, ..], _),
            ExprKind::Call(call_expr, [first_param, ..]),
        ) => {
            if let ExprKind::Path(QPath::Resolved(_, call_path)) = call_expr.kind {
                if has_identical_segments(path.segments, call_path.segments)
                    && has_same_non_ref_symbol(first_pat, first_param)
                {
                    return true;
                }
            }
        },
        // Example: `val => val`, or `ref val => *val`
        (PatKind::Binding(annot, _, pat_ident, _), _) => {
            let new_expr = if let (
                BindingAnnotation::Ref | BindingAnnotation::RefMut,
                ExprKind::Unary(UnOp::Deref, operand_expr),
            ) = (annot, &expr.kind)
            {
                operand_expr
            } else {
                expr
            };

            if let ExprKind::Path(QPath::Resolved(
                _,
                Path {
                    segments: [first_seg, ..],
                    ..
                },
            )) = new_expr.kind
            {
                return pat_ident.name == first_seg.ident.name;
            }
        },
        // Example: `Custom::TypeA => Custom::TypeB`, or `None => None`
        (PatKind::Path(QPath::Resolved(_, p_path)), ExprKind::Path(QPath::Resolved(_, e_path))) => {
            return has_identical_segments(p_path.segments, e_path.segments);
        },
        // Example: `5 => 5`
        (PatKind::Lit(pat_lit_expr), ExprKind::Lit(expr_spanned)) => {
            if let ExprKind::Lit(pat_spanned) = &pat_lit_expr.kind {
                return pat_spanned.node == expr_spanned.node;
            }
        },
        _ => {},
    }

    false
}

fn has_identical_segments(left_segs: &[PathSegment<'_>], right_segs: &[PathSegment<'_>]) -> bool {
    if left_segs.len() != right_segs.len() {
        return false;
    }
    for i in 0..left_segs.len() {
        if left_segs[i].ident.name != right_segs[i].ident.name {
            return false;
        }
    }
    true
}

fn has_same_non_ref_symbol(pat: &Pat<'_>, expr: &Expr<'_>) -> bool {
    if_chain! {
        if let PatKind::Binding(annot, _, pat_ident, _) = pat.kind;
        if !matches!(annot, BindingAnnotation::Ref | BindingAnnotation::RefMut);
        if let ExprKind::Path(QPath::Resolved(_, Path {segments: [first_seg, ..], .. })) = expr.kind;
        then {
            return pat_ident.name == first_seg.ident.name;
        }
    }

    false
}
