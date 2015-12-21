use rustc::lint::*;
use rustc_front::hir::*;
use syntax::codemap::Span;
use utils::{get_item_name, match_type, snippet, span_help_and_lint, walk_ptrs_ty};
use utils::HASHMAP_PATH;

/// **What it does:** This lint checks for uses of `contains_key` + `insert` on `HashMap`.
///
/// **Why is this bad?** Using `HashMap::entry` is more efficient.
///
/// **Known problems:** Hopefully none.
///
/// **Example:** `if !m.contains_key(&k) { m.insert(k, v) }`
declare_lint! {
    pub HASHMAP_ENTRY,
    Warn,
    "use of `contains_key` followed by `insert` on a `HashMap`"
}

#[derive(Copy,Clone)]
pub struct HashMapLint;

impl LintPass for HashMapLint {
    fn get_lints(&self) -> LintArray {
        lint_array!(HASHMAP_ENTRY)
    }
}

impl LateLintPass for HashMapLint {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if_let_chain! {
            [
                let ExprIf(ref check, ref then, _) = expr.node,
                let ExprUnary(UnOp::UnNot, ref check) = check.node,
                let ExprMethodCall(ref name, _, ref params) = check.node,
                params.len() >= 2,
                name.node.as_str() == "contains_key"
            ], {
                let map = &params[0];
                let obj_ty = walk_ptrs_ty(cx.tcx.expr_ty(map));

                if match_type(cx, obj_ty, &HASHMAP_PATH) {
                    if let Some(ref then) = then.expr {
                        check_for_insert(cx, expr.span, map, then);
                    }
                    else if then.stmts.len() == 1 {
                        if let StmtSemi(ref stmt, _) = then.stmts[0].node {
                            check_for_insert(cx, expr.span, map, stmt);
                        }
                    }
                }
            }
        }
    }
}

fn check_for_insert(cx: &LateContext, span: Span, map: &Expr, expr: &Expr) {
    if_let_chain! {
        [
            let ExprMethodCall(ref name, _, ref params) = expr.node,
            params.len() >= 3,
            name.node.as_str() == "insert",
            get_item_name(cx, map) == get_item_name(cx, &*params[0])
        ], {
            span_help_and_lint(cx, HASHMAP_ENTRY, span,
                               "usage of `contains_key` followed by `insert` on `HashMap`",
                               &format!("Consider using `{}.entry({}).or_insert({})`",
                                        snippet(cx, map.span, ".."),
                                        snippet(cx, params[1].span, ".."),
                                        snippet(cx, params[2].span, "..")));
        }
    }
}
