use super::NEEDLESS_MATCH;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::{is_type_diagnostic_item, same_type_and_consts};
use clippy_utils::{
    SpanlessEq, eq_expr_value, get_parent_expr_for_hir, higher, is_else_clause, is_res_lang_ctor, over, path_res,
    peel_blocks_with_stmt,
};
use rustc_errors::Applicability;
use rustc_hir::LangItem::OptionNone;
use rustc_hir::{
    Arm, BindingMode, ByRef, Expr, ExprKind, ItemKind, Node, Pat, PatExpr, PatExprKind, PatKind, Path, QPath,
};
use rustc_lint::LateContext;
use rustc_span::sym;

pub(crate) fn check_match(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    if arms.len() > 1 && expr_ty_matches_p_ty(cx, ex, expr) && check_all_arms(cx, ex, arms) {
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            NEEDLESS_MATCH,
            expr.span,
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
pub(crate) fn check_if_let<'tcx>(cx: &LateContext<'tcx>, ex: &Expr<'_>, if_let: &higher::IfLet<'tcx>) {
    if !is_else_clause(cx.tcx, ex) && expr_ty_matches_p_ty(cx, if_let.let_expr, ex) && check_if_let_inner(cx, if_let) {
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

fn check_all_arms(cx: &LateContext<'_>, match_expr: &Expr<'_>, arms: &[Arm<'_>]) -> bool {
    for arm in arms {
        let arm_expr = peel_blocks_with_stmt(arm.body);

        if let Some(guard_expr) = &arm.guard
            && guard_expr.can_have_side_effects()
        {
            return false;
        }

        if let PatKind::Wild = arm.pat.kind {
            if !eq_expr_value(cx, match_expr, strip_return(arm_expr)) {
                return false;
            }
        } else if !pat_same_as_expr(arm.pat, arm_expr) {
            return false;
        }
    }

    true
}

fn check_if_let_inner(cx: &LateContext<'_>, if_let: &higher::IfLet<'_>) -> bool {
    if let Some(if_else) = if_let.if_else {
        if !pat_same_as_expr(if_let.let_pat, peel_blocks_with_stmt(if_let.if_then)) {
            return false;
        }

        // Recursively check for each `else if let` phrase,
        if let Some(ref nested_if_let) = higher::IfLet::hir(cx, if_else)
            && SpanlessEq::new(cx).eq_expr(nested_if_let.let_expr, if_let.let_expr)
        {
            return check_if_let_inner(cx, nested_if_let);
        }

        if matches!(if_else.kind, ExprKind::Block(..)) {
            let else_expr = peel_blocks_with_stmt(if_else);
            if matches!(else_expr.kind, ExprKind::Block(..)) {
                return false;
            }
            let ret = strip_return(else_expr);
            let let_expr_ty = cx.typeck_results().expr_ty(if_let.let_expr);
            if is_type_diagnostic_item(cx, let_expr_ty, sym::Option) {
                return is_res_lang_ctor(cx, path_res(cx, ret), OptionNone) || eq_expr_value(cx, if_let.let_expr, ret);
            }
            return eq_expr_value(cx, if_let.let_expr, ret);
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

/// Manually check for coercion casting by checking if the type of the match operand or let expr
/// differs with the assigned local variable or the function return type.
fn expr_ty_matches_p_ty(cx: &LateContext<'_>, expr: &Expr<'_>, p_expr: &Expr<'_>) -> bool {
    match cx.tcx.parent_hir_node(p_expr.hir_id) {
        // Compare match_expr ty with local in `let local = match match_expr {..}`
        Node::LetStmt(local) => {
            let results = cx.typeck_results();
            return same_type_and_consts(results.node_type(local.hir_id), results.expr_ty(expr));
        },
        // compare match_expr ty with RetTy in `fn foo() -> RetTy`
        Node::Item(item) => {
            if let ItemKind::Fn { .. } = item.kind {
                let output = cx
                    .tcx
                    .fn_sig(item.owner_id)
                    .instantiate_identity()
                    .output()
                    .skip_binder();
                return same_type_and_consts(output, cx.typeck_results().expr_ty(expr));
            }
        },
        // check the parent expr for this whole block `{ match match_expr {..} }`
        Node::Block(block) => {
            if let Some(block_parent_expr) = get_parent_expr_for_hir(cx, block.hir_id) {
                return expr_ty_matches_p_ty(cx, expr, block_parent_expr);
            }
        },
        // recursively call on `if xxx {..}` etc.
        Node::Expr(p_expr) => {
            return expr_ty_matches_p_ty(cx, expr, p_expr);
        },
        _ => {},
    }
    false
}

fn pat_same_as_expr(pat: &Pat<'_>, expr: &Expr<'_>) -> bool {
    let expr = strip_return(expr);
    match (&pat.kind, &expr.kind) {
        // Example: `Some(val) => Some(val)`
        (PatKind::TupleStruct(QPath::Resolved(_, path), tuple_params, _), ExprKind::Call(call_expr, call_params)) => {
            if let ExprKind::Path(QPath::Resolved(_, call_path)) = call_expr.kind {
                return over(path.segments, call_path.segments, |pat_seg, call_seg| {
                    pat_seg.ident.name == call_seg.ident.name
                }) && same_non_ref_symbols(tuple_params, call_params);
            }
        },
        // Example: `val => val`
        (
            PatKind::Binding(annot, _, pat_ident, _),
            ExprKind::Path(QPath::Resolved(
                _,
                Path {
                    segments: [first_seg, ..],
                    ..
                },
            )),
        ) => {
            return !matches!(annot, BindingMode(ByRef::Yes(_), _)) && pat_ident.name == first_seg.ident.name;
        },
        // Example: `Custom::TypeA => Custom::TypeB`, or `None => None`
        (
            PatKind::Expr(PatExpr {
                kind: PatExprKind::Path(QPath::Resolved(_, p_path)),
                ..
            }),
            ExprKind::Path(QPath::Resolved(_, e_path)),
        ) => {
            return over(p_path.segments, e_path.segments, |p_seg, e_seg| {
                p_seg.ident.name == e_seg.ident.name
            });
        },
        // Example: `5 => 5`
        (PatKind::Expr(pat_expr_expr), ExprKind::Lit(expr_spanned)) => {
            if let PatExprKind::Lit {
                lit: pat_spanned,
                negated: false,
            } = &pat_expr_expr.kind
            {
                return pat_spanned.node == expr_spanned.node;
            }
        },
        _ => {},
    }

    false
}

fn same_non_ref_symbols(pats: &[Pat<'_>], exprs: &[Expr<'_>]) -> bool {
    if pats.len() != exprs.len() {
        return false;
    }

    for i in 0..pats.len() {
        if !pat_same_as_expr(&pats[i], &exprs[i]) {
            return false;
        }
    }

    true
}
