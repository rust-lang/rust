use rustc::hir;
use rustc::lint::*;
use utils::{span_lint_and_then, span_lint, snippet_opt, SpanlessEq, get_trait_def_id, implements_trait};

/// **What it does:** This lint checks for `+=` operations and similar
///
/// **Why is this bad?** Projects with many developers from languages without those operations
///                      may find them unreadable and not worth their weight
///
/// **Known problems:** Types implementing `OpAssign` don't necessarily implement `Op`
///
/// **Example:**
/// ```
/// a += 1;
/// ```
declare_restriction_lint! {
    pub ASSIGN_OPS,
    "Any assignment operation"
}

/// **What it does:** Check for `a = a op b` or `a = b commutative_op a` patterns
///
/// **Why is this bad?** These can be written as the shorter `a op= b`
///
/// **Known problems:** While forbidden by the spec, `OpAssign` traits may have implementations that differ from the regular `Op` impl
///
/// **Example:**
///
/// ```
/// let mut a = 5;
/// ...
/// a = a + b;
/// ```
declare_lint! {
    pub ASSIGN_OP_PATTERN,
    Warn,
    "assigning the result of an operation on a variable to that same variable"
}

#[derive(Copy, Clone, Default)]
pub struct AssignOps;

impl LintPass for AssignOps {
    fn get_lints(&self) -> LintArray {
        lint_array!(ASSIGN_OPS, ASSIGN_OP_PATTERN)
    }
}

impl LateLintPass for AssignOps {
    fn check_expr(&mut self, cx: &LateContext, expr: &hir::Expr) {
        match expr.node {
            hir::ExprAssignOp(op, ref lhs, ref rhs) => {
                if let (Some(l), Some(r)) = (snippet_opt(cx, lhs.span), snippet_opt(cx, rhs.span)) {
                    span_lint_and_then(cx, ASSIGN_OPS, expr.span, "assign operation detected", |db| {
                        match rhs.node {
                            hir::ExprBinary(op2, _, _) if op2 != op => {
                                db.span_suggestion(expr.span,
                                                   "replace it with",
                                                   format!("{} = {} {} ({})", l, l, op.node.as_str(), r));
                            }
                            _ => {
                                db.span_suggestion(expr.span,
                                                   "replace it with",
                                                   format!("{} = {} {} {}", l, l, op.node.as_str(), r));
                            }
                        }
                    });
                } else {
                    span_lint(cx, ASSIGN_OPS, expr.span, "assign operation detected");
                }
            }
            hir::ExprAssign(ref assignee, ref e) => {
                if let hir::ExprBinary(op, ref l, ref r) = e.node {
                    let lint = |assignee: &hir::Expr, rhs: &hir::Expr| {
                        let ty = cx.tcx.expr_ty(assignee);
                        if ty.walk_shallow().next().is_some() {
                            return; // implements_trait does not work with generics
                        }
                        let rty = cx.tcx.expr_ty(rhs);
                        if rty.walk_shallow().next().is_some() {
                            return; // implements_trait does not work with generics
                        }
                        macro_rules! ops {
                            ($op:expr, $cx:expr, $ty:expr, $rty:expr, $($trait_name:ident:$full_trait_name:ident),+) => {
                                match $op {
                                    $(hir::$full_trait_name => {
                                        let [krate, module] = ::utils::paths::OPS_MODULE;
                                        let path = [krate, module, concat!(stringify!($trait_name), "Assign")];
                                        let trait_id = if let Some(trait_id) = get_trait_def_id($cx, &path) {
                                            trait_id
                                        } else {
                                            return; // useless if the trait doesn't exist
                                        };
                                        implements_trait($cx, $ty, trait_id, vec![$rty])
                                    },)*
                                    _ => false,
                                }
                            }
                        }
                        if ops!(op.node,
                                cx,
                                ty,
                                rty,
                                Add: BiAdd,
                                Sub: BiSub,
                                Mul: BiMul,
                                Div: BiDiv,
                                Rem: BiRem,
                                And: BiAnd,
                                Or: BiOr,
                                BitAnd: BiBitAnd,
                                BitOr: BiBitOr,
                                BitXor: BiBitXor,
                                Shr: BiShr,
                                Shl: BiShl) {
                            if let (Some(snip_a), Some(snip_r)) = (snippet_opt(cx, assignee.span),
                                                                   snippet_opt(cx, rhs.span)) {
                                span_lint_and_then(cx,
                                                   ASSIGN_OP_PATTERN,
                                                   expr.span,
                                                   "manual implementation of an assign operation",
                                                   |db| {
                                                       db.span_suggestion(expr.span,
                                                                          "replace it with",
                                                                          format!("{} {}= {}", snip_a, op.node.as_str(), snip_r));
                                                   });
                            } else {
                                span_lint(cx,
                                          ASSIGN_OP_PATTERN,
                                          expr.span,
                                          "manual implementation of an assign operation");
                            }
                        }
                    };
                    // a = a op b
                    if SpanlessEq::new(cx).ignore_fn().eq_expr(assignee, l) {
                        lint(assignee, r);
                    }
                    // a = b commutative_op a
                    if SpanlessEq::new(cx).ignore_fn().eq_expr(assignee, r) {
                        match op.node {
                            hir::BiAdd | hir::BiMul | hir::BiAnd | hir::BiOr | hir::BiBitXor | hir::BiBitAnd |
                            hir::BiBitOr => {
                                lint(assignee, l);
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
