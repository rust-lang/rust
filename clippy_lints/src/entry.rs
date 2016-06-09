use rustc::hir::*;
use rustc::hir::intravisit::{Visitor, walk_expr, walk_block};
use rustc::lint::*;
use syntax::codemap::Span;
use utils::SpanlessEq;
use utils::{get_item_name, match_type, paths, snippet, span_lint_and_then, walk_ptrs_ty};

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
        if let ExprIf(ref check, ref then_block, ref else_block) = expr.node {
            if let ExprUnary(UnOp::UnNot, ref check) = check.node {
                if let Some((ty, map, key)) = check_cond(cx, check) {
                    // in case of `if !m.contains_key(&k) { m.insert(k, v); }`
                    // we can give a better error message
                    let sole_expr = else_block.is_none() &&
                                    ((then_block.expr.is_some() as usize) + then_block.stmts.len() == 1);

                    let mut visitor = InsertVisitor {
                        cx: cx,
                        span: expr.span,
                        ty: ty,
                        map: map,
                        key: key,
                        sole_expr: sole_expr,
                    };

                    walk_block(&mut visitor, then_block);
                }
            } else if let Some(ref else_block) = *else_block {
                if let Some((ty, map, key)) = check_cond(cx, check) {
                    let mut visitor = InsertVisitor {
                        cx: cx,
                        span: expr.span,
                        ty: ty,
                        map: map,
                        key: key,
                        sole_expr: false,
                    };

                    walk_expr(&mut visitor, else_block);
                }
            }
        }
    }
}

fn check_cond<'a, 'tcx, 'b>(cx: &'a LateContext<'a, 'tcx>, check: &'b Expr) -> Option<(&'static str, &'b Expr, &'b Expr)> {
    if_let_chain! {[
        let ExprMethodCall(ref name, _, ref params) = check.node,
        params.len() >= 2,
        name.node.as_str() == "contains_key",
        let ExprAddrOf(_, ref key) = params[1].node
    ], {
        let map = &params[0];
        let obj_ty = walk_ptrs_ty(cx.tcx.expr_ty(map));

        return if match_type(cx, obj_ty, &paths::BTREEMAP) {
            Some(("BTreeMap", map, key))
        }
        else if match_type(cx, obj_ty, &paths::HASHMAP) {
            Some(("HashMap", map, key))
        }
        else {
            None
        };
    }}

    None
}

struct InsertVisitor<'a, 'tcx: 'a, 'b> {
    cx: &'a LateContext<'a, 'tcx>,
    span: Span,
    ty: &'static str,
    map: &'b Expr,
    key: &'b Expr,
    sole_expr: bool,
}

impl<'a, 'tcx, 'v, 'b> Visitor<'v> for InsertVisitor<'a, 'tcx, 'b> {
    fn visit_expr(&mut self, expr: &'v Expr) {
        if_let_chain! {[
            let ExprMethodCall(ref name, _, ref params) = expr.node,
            params.len() == 3,
            name.node.as_str() == "insert",
            get_item_name(self.cx, self.map) == get_item_name(self.cx, &*params[0]),
            SpanlessEq::new(self.cx).eq_expr(self.key, &params[1])
        ], {
            span_lint_and_then(self.cx, MAP_ENTRY, self.span,
                               &format!("usage of `contains_key` followed by `insert` on `{}`", self.ty), |db| {
                if self.sole_expr {
                    let help = format!("{}.entry({}).or_insert({})",
                                       snippet(self.cx, self.map.span, "map"),
                                       snippet(self.cx, params[1].span, ".."),
                                       snippet(self.cx, params[2].span, ".."));

                    db.span_suggestion(self.span, "Consider using", help);
                }
                else {
                    let help = format!("Consider using `{}.entry({})`",
                                       snippet(self.cx, self.map.span, "map"),
                                       snippet(self.cx, params[1].span, ".."));

                    db.span_note(self.span, &help);
                }
            });
        }}

        if !self.sole_expr {
            walk_expr(self, expr);
        }
    }
}
