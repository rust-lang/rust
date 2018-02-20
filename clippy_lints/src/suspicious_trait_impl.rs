use rustc::lint::*;
use rustc::hir;
use rustc::hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use syntax::ast;
use utils::{get_trait_def_id, span_lint};

/// **What it does:** Lints for suspicious operations in impls of arithmetic operators, e.g.
/// subtracting elements in an Add impl.
///
/// **Why this is bad?** This is probably a typo or copy-and-paste error and not intended.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// impl Add for Foo {
///     type Output = Foo;
///
///     fn add(self, other: Foo) -> Foo {
///         Foo(self.0 - other.0)
///     }
/// }
/// ```
declare_lint! {
    pub SUSPICIOUS_ARITHMETIC_IMPL,
    Warn,
    "suspicious use of operators in impl of arithmetic trait"
}

/// **What it does:** Lints for suspicious operations in impls of OpAssign, e.g.
/// subtracting elements in an AddAssign impl.
///
/// **Why this is bad?** This is probably a typo or copy-and-paste error and not intended.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// impl AddAssign for Foo {
///     fn add_assign(&mut self, other: Foo) {
///         *self = *self - other;
///     }
/// }
/// ```
declare_lint! {
    pub SUSPICIOUS_OP_ASSIGN_IMPL,
    Warn,
    "suspicious use of operators in impl of OpAssign trait"
}

#[derive(Copy, Clone)]
pub struct SuspiciousImpl;

impl LintPass for SuspiciousImpl {
    fn get_lints(&self) -> LintArray {
        lint_array![SUSPICIOUS_ARITHMETIC_IMPL, SUSPICIOUS_OP_ASSIGN_IMPL]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for SuspiciousImpl {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        use rustc::hir::BinOp_::*;
        if let hir::ExprBinary(binop, _, _) = expr.node {
            // Check if the binary expression is part of another binary expression
            // as a child node
            let mut parent_expr = cx.tcx.hir.get_parent_node(expr.id);
            while parent_expr != ast::CRATE_NODE_ID {
                if_chain! {
                    if let hir::map::Node::NodeExpr(e) = cx.tcx.hir.get(parent_expr);
                    if let hir::ExprBinary(_, _, _) = e.node;
                    then {
                        return
                    }
                }

                parent_expr = cx.tcx.hir.get_parent_node(parent_expr);
            }
            // as a parent node
            let mut visitor = BinaryExprVisitor {
                in_binary_expr: false,
            };
            walk_expr(&mut visitor, expr);

            if visitor.in_binary_expr {
                return;
            }

            if let Some(impl_trait) = check_binop(
                cx,
                expr,
                &binop.node,
                &["Add", "Sub", "Mul", "Div"],
                &[BiAdd, BiSub, BiMul, BiDiv],
            ) {
                span_lint(
                    cx,
                    SUSPICIOUS_ARITHMETIC_IMPL,
                    binop.span,
                    &format!(
                        r#"Suspicious use of binary operator in `{}` impl"#,
                        impl_trait
                    ),
                );
            }

            if let Some(impl_trait) = check_binop(
                cx,
                expr,
                &binop.node,
                &[
                    "AddAssign",
                    "SubAssign",
                    "MulAssign",
                    "DivAssign",
                    "BitAndAssign",
                    "BitOrAssign",
                    "BitXorAssign",
                    "RemAssign",
                    "ShlAssign",
                    "ShrAssign",
                ],
                &[
                    BiAdd, BiSub, BiMul, BiDiv, BiBitAnd, BiBitOr, BiBitXor, BiRem, BiShl, BiShr
                ],
            ) {
                span_lint(
                    cx,
                    SUSPICIOUS_OP_ASSIGN_IMPL,
                    binop.span,
                    &format!(
                        r#"Suspicious use of binary operator in `{}` impl"#,
                        impl_trait
                    ),
                );
            }
        }
    }
}

fn check_binop<'a>(
    cx: &LateContext,
    expr: &hir::Expr,
    binop: &hir::BinOp_,
    traits: &[&'a str],
    expected_ops: &[hir::BinOp_],
) -> Option<&'a str> {
    let mut trait_ids = vec![];
    let [krate, module] = ::utils::paths::OPS_MODULE;

    for t in traits {
        let path = [krate, module, t];
        if let Some(trait_id) = get_trait_def_id(cx, &path) {
            trait_ids.push(trait_id);
        } else {
            return None;
        }
    }

    // Get the actually implemented trait
    let parent_fn = cx.tcx.hir.get_parent(expr.id);
    let parent_impl = cx.tcx.hir.get_parent(parent_fn);

    if_chain! {
        if parent_impl != ast::CRATE_NODE_ID;
        if let hir::map::Node::NodeItem(item) = cx.tcx.hir.get(parent_impl);
        if let hir::Item_::ItemImpl(_, _, _, _, Some(ref trait_ref), _, _) = item.node;
        if let Some(idx) = trait_ids.iter().position(|&tid| tid == trait_ref.path.def.def_id());
        if *binop != expected_ops[idx];
        then{
            return Some(traits[idx])
        }
    }

    None
}

struct BinaryExprVisitor {
    in_binary_expr: bool,
}

impl<'a, 'tcx: 'a> Visitor<'tcx> for BinaryExprVisitor {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        if let hir::ExprBinary(_, _, _) = expr.node {
            self.in_binary_expr = true;
        }

        walk_expr(self, expr);
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}
