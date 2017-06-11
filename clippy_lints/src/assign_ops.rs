use rustc::hir;
use rustc::lint::*;
use syntax::ast;
use utils::{span_lint_and_then, snippet_opt, SpanlessEq, get_trait_def_id, implements_trait};
use utils::{higher, sugg};

/// **What it does:** Checks for compound assignment operations (`+=` and similar).
///
/// **Why is this bad?** Projects with many developers from languages without
/// those operations may find them unreadable and not worth their weight.
///
/// **Known problems:** Types implementing `OpAssign` don't necessarily implement `Op`.
///
/// **Example:**
/// ```rust
/// a += 1;
/// ```
declare_restriction_lint! {
    pub ASSIGN_OPS,
    "any compound assignment operation"
}

/// **What it does:** Checks for `a = a op b` or `a = b commutative_op a` patterns.
///
/// **Why is this bad?** These can be written as the shorter `a op= b`.
///
/// **Known problems:** While forbidden by the spec, `OpAssign` traits may have
/// implementations that differ from the regular `Op` impl.
///
/// **Example:**
/// ```rust
/// let mut a = 5;
/// ...
/// a = a + b;
/// ```
declare_lint! {
    pub ASSIGN_OP_PATTERN,
    Warn,
    "assigning the result of an operation on a variable to that same variable"
}

/// **What it does:** Checks for `a op= a op b` or `a op= b op a` patterns.
///
/// **Why is this bad?** Most likely these are bugs where one meant to write `a op= b`.
///
/// **Known problems:** Someone might actually mean `a op= a op b`, but that
/// should rather be written as `a = (2 * a) op b` where applicable.
///
/// **Example:**
/// ```rust
/// let mut a = 5;
/// ...
/// a += a + b;
/// ```
declare_lint! {
    pub MISREFACTORED_ASSIGN_OP,
    Warn,
    "having a variable on both sides of an assign op"
}

#[derive(Copy, Clone, Default)]
pub struct AssignOps;

impl LintPass for AssignOps {
    fn get_lints(&self) -> LintArray {
        lint_array!(ASSIGN_OPS, ASSIGN_OP_PATTERN, MISREFACTORED_ASSIGN_OP)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for AssignOps {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprAssignOp(op, ref lhs, ref rhs) => {
                span_lint_and_then(cx, ASSIGN_OPS, expr.span, "assign operation detected", |db| {
                    let lhs = &sugg::Sugg::hir(cx, lhs, "..");
                    let rhs = &sugg::Sugg::hir(cx, rhs, "..");

                    db.span_suggestion(expr.span,
                                       "replace it with",
                                       format!("{} = {}", lhs, sugg::make_binop(higher::binop(op.node), lhs, rhs)));
                });
                if let hir::ExprBinary(binop, ref l, ref r) = rhs.node {
                    if op.node == binop.node {
                        let lint = |assignee: &hir::Expr, rhs: &hir::Expr| {
                            span_lint_and_then(cx,
                                               MISREFACTORED_ASSIGN_OP,
                                               expr.span,
                                               "variable appears on both sides of an assignment operation",
                                               |db| if let (Some(snip_a), Some(snip_r)) =
                                                   (snippet_opt(cx, assignee.span), snippet_opt(cx, rhs.span)) {
                                                   db.span_suggestion(expr.span,
                                                                      "replace it with",
                                                                      format!("{} {}= {}",
                                                                              snip_a,
                                                                              op.node.as_str(),
                                                                              snip_r));
                                               });
                        };
                        // lhs op= l op r
                        if SpanlessEq::new(cx).ignore_fn().eq_expr(lhs, l) {
                            lint(lhs, r);
                        }
                        // lhs op= l commutative_op r
                        if is_commutative(op.node) && SpanlessEq::new(cx).ignore_fn().eq_expr(lhs, r) {
                            lint(lhs, l);
                        }
                    }
                }
            },
            hir::ExprAssign(ref assignee, ref e) => {
                if let hir::ExprBinary(op, ref l, ref r) = e.node {
                    #[allow(cyclomatic_complexity)]
                    let lint = |assignee: &hir::Expr, rhs: &hir::Expr| {
                        let ty = cx.tables.expr_ty(assignee);
                        let rty = cx.tables.expr_ty(rhs);
                        macro_rules! ops {
                            ($op:expr,
                             $cx:expr,
                             $ty:expr,
                             $rty:expr,
                             $($trait_name:ident:$full_trait_name:ident),+) => {
                                match $op {
                                    $(hir::$full_trait_name => {
                                        let [krate, module] = ::utils::paths::OPS_MODULE;
                                        let path = [krate, module, concat!(stringify!($trait_name), "Assign")];
                                        let trait_id = if let Some(trait_id) = get_trait_def_id($cx, &path) {
                                            trait_id
                                        } else {
                                            return; // useless if the trait doesn't exist
                                        };
                                        // check that we are not inside an `impl AssignOp` of this exact operation
                                        let parent_fn = cx.tcx.hir.get_parent(e.id);
                                        let parent_impl = cx.tcx.hir.get_parent(parent_fn);
                                        // the crate node is the only one that is not in the map
                                        if_let_chain!{[
                                            parent_impl != ast::CRATE_NODE_ID,
                                            let hir::map::Node::NodeItem(item) = cx.tcx.hir.get(parent_impl),
                                            let hir::Item_::ItemImpl(_, _, _, _, Some(ref trait_ref), _, _) = item.node,
                                            trait_ref.path.def.def_id() == trait_id
                                        ], { return; }}
                                        implements_trait($cx, $ty, trait_id, &[$rty])
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
                            span_lint_and_then(cx,
                                               ASSIGN_OP_PATTERN,
                                               expr.span,
                                               "manual implementation of an assign operation",
                                               |db| if let (Some(snip_a), Some(snip_r)) =
                                                   (snippet_opt(cx, assignee.span), snippet_opt(cx, rhs.span)) {
                                                   db.span_suggestion(expr.span,
                                                                      "replace it with",
                                                                      format!("{} {}= {}",
                                                                              snip_a,
                                                                              op.node.as_str(),
                                                                              snip_r));
                                               });
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
                            },
                            _ => {},
                        }
                    }
                }
            },
            _ => {},
        }
    }
}

fn is_commutative(op: hir::BinOp_) -> bool {
    use rustc::hir::BinOp_::*;
    match op {
        BiAdd | BiMul | BiAnd | BiOr | BiBitXor | BiBitAnd | BiBitOr | BiEq | BiNe => true,
        BiSub | BiDiv | BiRem | BiShl | BiShr | BiLt | BiLe | BiGe | BiGt => false,
    }
}
