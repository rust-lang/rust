use crate::utils::SpanlessEq;
use crate::utils::{get_item_name, higher, match_type, paths, snippet, span_lint_and_then, walk_ptrs_ty};
use if_chain::if_chain;
use rustc::hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for uses of `contains_key` + `insert` on `HashMap`
    /// or `BTreeMap`.
    ///
    /// **Why is this bad?** Using `entry` is more efficient.
    ///
    /// **Known problems:** Some false negatives, eg.:
    /// ```rust
    /// let k = &key;
    /// if !m.contains_key(k) {
    ///     m.insert(k.clone(), v);
    /// }
    /// ```
    ///
    /// **Example:**
    /// ```rust
    /// if !m.contains_key(&k) {
    ///     m.insert(k, v)
    /// }
    /// ```
    /// can be rewritten as:
    /// ```rust
    /// m.entry(k).or_insert(v);
    /// ```
    pub MAP_ENTRY,
    perf,
    "use of `contains_key` followed by `insert` on a `HashMap` or `BTreeMap`"
}

declare_lint_pass!(HashMapPass => [MAP_ENTRY]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for HashMapPass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let Some((ref check, ref then_block, ref else_block)) = higher::if_block(&expr) {
            if let ExprKind::Unary(UnOp::UnNot, ref check) = check.node {
                if let Some((ty, map, key)) = check_cond(cx, check) {
                    // in case of `if !m.contains_key(&k) { m.insert(k, v); }`
                    // we can give a better error message
                    let sole_expr = {
                        else_block.is_none()
                            && if let ExprKind::Block(ref then_block, _) = then_block.node {
                                (then_block.expr.is_some() as usize) + then_block.stmts.len() == 1
                            } else {
                                true
                            }
                    };

                    let mut visitor = InsertVisitor {
                        cx,
                        span: expr.span,
                        ty,
                        map,
                        key,
                        sole_expr,
                    };

                    walk_expr(&mut visitor, &**then_block);
                }
            } else if let Some(ref else_block) = *else_block {
                if let Some((ty, map, key)) = check_cond(cx, check) {
                    let mut visitor = InsertVisitor {
                        cx,
                        span: expr.span,
                        ty,
                        map,
                        key,
                        sole_expr: false,
                    };

                    walk_expr(&mut visitor, else_block);
                }
            }
        }
    }
}

fn check_cond<'a, 'tcx, 'b>(
    cx: &'a LateContext<'a, 'tcx>,
    check: &'b Expr,
) -> Option<(&'static str, &'b Expr, &'b Expr)> {
    if_chain! {
        if let ExprKind::MethodCall(ref path, _, ref params) = check.node;
        if params.len() >= 2;
        if path.ident.name == sym!(contains_key);
        if let ExprKind::AddrOf(_, ref key) = params[1].node;
        then {
            let map = &params[0];
            let obj_ty = walk_ptrs_ty(cx.tables.expr_ty(map));

            return if match_type(cx, obj_ty, &paths::BTREEMAP) {
                Some(("BTreeMap", map, key))
            }
            else if match_type(cx, obj_ty, &paths::HASHMAP) {
                Some(("HashMap", map, key))
            }
            else {
                None
            };
        }
    }

    None
}

struct InsertVisitor<'a, 'tcx, 'b> {
    cx: &'a LateContext<'a, 'tcx>,
    span: Span,
    ty: &'static str,
    map: &'b Expr,
    key: &'b Expr,
    sole_expr: bool,
}

impl<'a, 'tcx, 'b> Visitor<'tcx> for InsertVisitor<'a, 'tcx, 'b> {
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        if_chain! {
            if let ExprKind::MethodCall(ref path, _, ref params) = expr.node;
            if params.len() == 3;
            if path.ident.name == sym!(insert);
            if get_item_name(self.cx, self.map) == get_item_name(self.cx, &params[0]);
            if SpanlessEq::new(self.cx).eq_expr(self.key, &params[1]);
            then {
                span_lint_and_then(self.cx, MAP_ENTRY, self.span,
                                   &format!("usage of `contains_key` followed by `insert` on a `{}`", self.ty), |db| {
                    if self.sole_expr {
                        let help = format!("{}.entry({}).or_insert({})",
                                           snippet(self.cx, self.map.span, "map"),
                                           snippet(self.cx, params[1].span, ".."),
                                           snippet(self.cx, params[2].span, ".."));

                        db.span_suggestion(
                            self.span,
                            "consider using",
                            help,
                            Applicability::MachineApplicable, // snippet
                        );
                    }
                    else {
                        let help = format!("{}.entry({})",
                                           snippet(self.cx, self.map.span, "map"),
                                           snippet(self.cx, params[1].span, ".."));

                        db.span_suggestion(
                            self.span,
                            "consider using",
                            help,
                            Applicability::MachineApplicable, // snippet
                        );
                    }
                });
            }
        }

        if !self.sole_expr {
            walk_expr(self, expr);
        }
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}
