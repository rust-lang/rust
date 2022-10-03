use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_res_lang_ctor, path_res, path_to_local_id};
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::LangItem::OptionSome;
use rustc_hir::{Arm, Block, BlockCheckMode, Expr, ExprKind, HirId, Pat, PatKind, UnsafeSource};
use rustc_lint::LateContext;
use rustc_span::{sym, SyntaxContext};

use super::manual_map::{check_with, SomeExpr};
use super::MANUAL_FILTER;

#[derive(Default)]
struct NeedsUnsafeBlock(pub bool);

impl<'tcx> Visitor<'tcx> for NeedsUnsafeBlock {
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        match expr.kind {
            ExprKind::Block(
                Block {
                    rules: BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided),
                    ..
                },
                _,
            ) => {
                self.0 = true;
            },
            _ => walk_expr(self, expr),
        }
    }
}

// Function called on the `expr` of `[&+]Some((ref | ref mut) x) => <expr>`
// Need to check if it's of the `if <cond> {<then_expr>} else {<else_expr>}`
// AND that only one `then/else_expr` resolves to `Some(x)` while the other resolves to `None`
// return `cond` if
fn get_cond_expr<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &Pat<'_>,
    expr: &'tcx Expr<'_>,
    ctxt: SyntaxContext,
) -> Option<SomeExpr<'tcx>> {
    if_chain! {
        if let Some(block_expr) = peels_blocks_incl_unsafe_opt(expr);
        if let ExprKind::If(cond, then_expr, Some(else_expr)) = block_expr.kind;
        if let PatKind::Binding(_,target, ..) = pat.kind;
        if let (then_visitor, else_visitor)
            = (handle_if_or_else_expr(cx, target, ctxt, then_expr),
                handle_if_or_else_expr(cx, target, ctxt, else_expr));
        if then_visitor != else_visitor; // check that one expr resolves to `Some(x)`, the other to `None`
        then {
            let mut needs_unsafe_block = NeedsUnsafeBlock::default();
            needs_unsafe_block.visit_expr(expr);
            return Some(SomeExpr {
                    expr: peels_blocks_incl_unsafe(cond.peel_drop_temps()),
                    needs_unsafe_block: needs_unsafe_block.0,
                    needs_negated: !then_visitor // if the `then_expr` resolves to `None`, need to negate the cond
                })
            }
    };
    None
}

fn peels_blocks_incl_unsafe_opt<'a>(expr: &'a Expr<'a>) -> Option<&'a Expr<'a>> {
    // we don't want to use `peel_blocks` here because we don't care if the block is unsafe, it's
    // checked by `NeedsUnsafeBlock`
    if let ExprKind::Block(block, None) = expr.kind {
        if block.stmts.is_empty() {
            return block.expr;
        }
    };
    None
}

fn peels_blocks_incl_unsafe<'a>(expr: &'a Expr<'a>) -> &'a Expr<'a> {
    peels_blocks_incl_unsafe_opt(expr).unwrap_or(expr)
}

// function called for each <ifelse> expression:
// Some(x) => if <cond> {
//    <ifelse>
// } else {
//    <ifelse>
// }
// Returns true if <ifelse> resolves to `Some(x)`, `false` otherwise
fn handle_if_or_else_expr<'tcx>(
    cx: &LateContext<'_>,
    target: HirId,
    ctxt: SyntaxContext,
    if_or_else_expr: &'tcx Expr<'_>,
) -> bool {
    if let Some(inner_expr) = peels_blocks_incl_unsafe_opt(if_or_else_expr) {
        // there can be not statements in the block as they would be removed when switching to `.filter`
        if let ExprKind::Call(callee, [arg]) = inner_expr.kind {
            return ctxt == if_or_else_expr.span.ctxt()
                && is_res_lang_ctor(cx, path_res(cx, callee), OptionSome)
                && path_to_local_id(arg, target);
        }
    };
    false
}

// given the closure: `|<pattern>| <expr>`
// returns `|&<pattern>| <expr>`
fn add_ampersand_if_copy(body_str: String, has_copy_trait: bool) -> String {
    if has_copy_trait {
        let mut with_ampersand = body_str;
        with_ampersand.insert(1, '&');
        with_ampersand
    } else {
        body_str
    }
}

pub(super) fn check_match<'tcx>(
    cx: &LateContext<'tcx>,
    scrutinee: &'tcx Expr<'_>,
    arms: &'tcx [Arm<'_>],
    expr: &'tcx Expr<'_>,
) {
    let ty = cx.typeck_results().expr_ty(expr);
    if_chain! {
        if is_type_diagnostic_item(cx, ty, sym::Option);
        if arms.len() == 2;
        if arms[0].guard.is_none();
        if arms[1].guard.is_none();
        then {
            check(cx, expr, scrutinee, arms[0].pat, arms[0].body, Some(arms[1].pat), arms[1].body)
        }
    }
}

pub(super) fn check_if_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    let_pat: &'tcx Pat<'_>,
    let_expr: &'tcx Expr<'_>,
    then_expr: &'tcx Expr<'_>,
    else_expr: &'tcx Expr<'_>,
) {
    check(cx, expr, let_expr, let_pat, then_expr, None, else_expr);
}

fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    scrutinee: &'tcx Expr<'_>,
    then_pat: &'tcx Pat<'_>,
    then_body: &'tcx Expr<'_>,
    else_pat: Option<&'tcx Pat<'_>>,
    else_body: &'tcx Expr<'_>,
) {
    if let Some(sugg_info) = check_with(
        cx,
        expr,
        scrutinee,
        then_pat,
        then_body,
        else_pat,
        else_body,
        get_cond_expr,
    ) {
        let body_str = add_ampersand_if_copy(sugg_info.body_str, sugg_info.scrutinee_impl_copy);
        span_lint_and_sugg(
            cx,
            MANUAL_FILTER,
            expr.span,
            "manual implementation of `Option::filter`",
            "try this",
            if sugg_info.needs_brackets {
                format!(
                    "{{ {}{}.filter({body_str}) }}",
                    sugg_info.scrutinee_str, sugg_info.as_ref_str
                )
            } else {
                format!("{}{}.filter({body_str})", sugg_info.scrutinee_str, sugg_info.as_ref_str)
            },
            sugg_info.app,
        );
    }
}
