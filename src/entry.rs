use rustc::lint::*;
use rustc_front::hir::*;
use syntax::codemap::Span;
use utils::{get_item_name, is_exp_equal, match_type, snippet, span_lint_and_then, walk_ptrs_ty};
use utils::{BTREEMAP_PATH, HASHMAP_PATH};

/// **What it does:** This lint checks for uses of `contains_key` + `insert` on `HashMap` or
/// `BTreeMap`.
///
/// **Why is this bad?** Using `entry` is more efficient.
///
/// **Known problems:** Some false negatives, eg.:
/// ```
/// let k = &key;
/// if !m.contains_key(k) { m.insert(k.clone(), v); }
/// ```
///
/// **Example:**
/// ```rust
/// if !m.contains_key(&k) { m.insert(k, v) }
/// ```
/// can be rewritten as:
/// ```rust
/// m.entry(k).or_insert(v);
/// ```
declare_lint! {
    pub MAP_ENTRY,
    Warn,
    "use of `contains_key` followed by `insert` on a `HashMap` or `BTreeMap`"
}

#[derive(Copy,Clone)]
pub struct HashMapLint;

impl LintPass for HashMapLint {
    fn get_lints(&self) -> LintArray {
        lint_array!(MAP_ENTRY)
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
                let key = match params[1].node {
                    ExprAddrOf(_, ref key) => key,
                    _ => return
                };

                let map = &params[0];
                let obj_ty = walk_ptrs_ty(cx.tcx.expr_ty(map));

                let kind = if match_type(cx, obj_ty, &BTREEMAP_PATH) {
                    "BTreeMap"
                }
                else if match_type(cx, obj_ty, &HASHMAP_PATH) {
                    "HashMap"
                }
                else {
                    return
                };

                let sole_expr = if then.expr.is_some() { 1 } else { 0 } + then.stmts.len() == 1;

                if let Some(ref then) = then.expr {
                    check_for_insert(cx, expr.span, map, key, then, sole_expr, kind);
                }

                for stmt in &then.stmts {
                    if let StmtSemi(ref stmt, _) = stmt.node {
                        check_for_insert(cx, expr.span, map, key, stmt, sole_expr, kind);
                    }
                }
            }
        }
    }
}

fn check_for_insert(cx: &LateContext, span: Span, map: &Expr, key: &Expr, expr: &Expr, sole_expr: bool, kind: &str) {
    if_let_chain! {
        [
            let ExprMethodCall(ref name, _, ref params) = expr.node,
            params.len() == 3,
            name.node.as_str() == "insert",
            get_item_name(cx, map) == get_item_name(cx, &*params[0]),
            is_exp_equal(cx, key, &params[1])
        ], {
            let help = if sole_expr {
                format!("{}.entry({}).or_insert({})",
                        snippet(cx, map.span, ".."),
                        snippet(cx, params[1].span, ".."),
                        snippet(cx, params[2].span, ".."))
            }
            else {
                format!("{}.entry({})",
                        snippet(cx, map.span, ".."),
                        snippet(cx, params[1].span, ".."))
            };

            span_lint_and_then(cx, MAP_ENTRY, span,
                               &format!("usage of `contains_key` followed by `insert` on `{}`", kind), |db| {
                db.span_suggestion(span, "Consider using", help.clone());
            });
        }
    }
}
