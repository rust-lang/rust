use crate::utils::{get_trait_def_id, span_lint, trait_ref_of_method};
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Lints for suspicious operations in impls of arithmetic operators, e.g.
    /// subtracting elements in an Add impl.
    ///
    /// **Why this is bad?** This is probably a typo or copy-and-paste error and not intended.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// impl Add for Foo {
    ///     type Output = Foo;
    ///
    ///     fn add(self, other: Foo) -> Foo {
    ///         Foo(self.0 - other.0)
    ///     }
    /// }
    /// ```
    pub SUSPICIOUS_ARITHMETIC_IMPL,
    correctness,
    "suspicious use of operators in impl of arithmetic trait"
}

declare_clippy_lint! {
    /// **What it does:** Lints for suspicious operations in impls of OpAssign, e.g.
    /// subtracting elements in an AddAssign impl.
    ///
    /// **Why this is bad?** This is probably a typo or copy-and-paste error and not intended.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// impl AddAssign for Foo {
    ///     fn add_assign(&mut self, other: Foo) {
    ///         *self = *self - other;
    ///     }
    /// }
    /// ```
    pub SUSPICIOUS_OP_ASSIGN_IMPL,
    correctness,
    "suspicious use of operators in impl of OpAssign trait"
}

declare_lint_pass!(SuspiciousImpl => [SUSPICIOUS_ARITHMETIC_IMPL, SUSPICIOUS_OP_ASSIGN_IMPL]);

impl<'tcx> LateLintPass<'tcx> for SuspiciousImpl {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if let hir::ExprKind::Binary(binop, _, _) | hir::ExprKind::AssignOp(binop, ..) = expr.kind {
            match binop.node {
                hir::BinOpKind::Eq
                | hir::BinOpKind::Lt
                | hir::BinOpKind::Le
                | hir::BinOpKind::Ne
                | hir::BinOpKind::Ge
                | hir::BinOpKind::Gt => return,
                _ => {},
            }

            // Check for more than one binary operation in the implemented function
            // Linting when multiple operations are involved can result in false positives
            if_chain! {
                let parent_fn = cx.tcx.hir().get_parent_item(expr.hir_id);
                if let hir::Node::ImplItem(impl_item) = cx.tcx.hir().get(parent_fn);
                if let hir::ImplItemKind::Fn(_, body_id) = impl_item.kind;
                let body = cx.tcx.hir().body(body_id);
                let mut visitor = BinaryExprVisitor { nb_binops: 0 };

                then {
                    walk_expr(&mut visitor, &body.value);
                    if visitor.nb_binops > 1 {
                        return;
                    }
                }
            }

            if let Some(impl_trait) = check_binop(
                cx,
                expr,
                binop.node,
                &[
                    "Add", "Sub", "Mul", "Div", "Rem", "BitAnd", "BitOr", "BitXor", "Shl", "Shr",
                ],
                &[
                    hir::BinOpKind::Add,
                    hir::BinOpKind::Sub,
                    hir::BinOpKind::Mul,
                    hir::BinOpKind::Div,
                    hir::BinOpKind::Rem,
                    hir::BinOpKind::BitAnd,
                    hir::BinOpKind::BitOr,
                    hir::BinOpKind::BitXor,
                    hir::BinOpKind::Shl,
                    hir::BinOpKind::Shr,
                ],
            ) {
                span_lint(
                    cx,
                    SUSPICIOUS_ARITHMETIC_IMPL,
                    binop.span,
                    &format!("suspicious use of binary operator in `{}` impl", impl_trait),
                );
            }

            if let Some(impl_trait) = check_binop(
                cx,
                expr,
                binop.node,
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
                    hir::BinOpKind::Add,
                    hir::BinOpKind::Sub,
                    hir::BinOpKind::Mul,
                    hir::BinOpKind::Div,
                    hir::BinOpKind::BitAnd,
                    hir::BinOpKind::BitOr,
                    hir::BinOpKind::BitXor,
                    hir::BinOpKind::Rem,
                    hir::BinOpKind::Shl,
                    hir::BinOpKind::Shr,
                ],
            ) {
                span_lint(
                    cx,
                    SUSPICIOUS_OP_ASSIGN_IMPL,
                    binop.span,
                    &format!("suspicious use of binary operator in `{}` impl", impl_trait),
                );
            }
        }
    }
}

fn check_binop(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    binop: hir::BinOpKind,
    traits: &[&'static str],
    expected_ops: &[hir::BinOpKind],
) -> Option<&'static str> {
    let mut trait_ids = vec![];
    let [krate, module] = crate::utils::paths::OPS_MODULE;

    for &t in traits {
        let path = [krate, module, t];
        if let Some(trait_id) = get_trait_def_id(cx, &path) {
            trait_ids.push(trait_id);
        } else {
            return None;
        }
    }

    // Get the actually implemented trait
    let parent_fn = cx.tcx.hir().get_parent_item(expr.hir_id);

    if_chain! {
        if let Some(trait_ref) = trait_ref_of_method(cx, parent_fn);
        if let Some(idx) = trait_ids.iter().position(|&tid| tid == trait_ref.path.res.def_id());
        if binop != expected_ops[idx];
        then{
            return Some(traits[idx])
        }
    }

    None
}

struct BinaryExprVisitor {
    nb_binops: u32,
}

impl<'tcx> Visitor<'tcx> for BinaryExprVisitor {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'_>) {
        match expr.kind {
            hir::ExprKind::Binary(..)
            | hir::ExprKind::Unary(hir::UnOp::Not | hir::UnOp::Neg, _)
            | hir::ExprKind::AssignOp(..) => self.nb_binops += 1,
            _ => {},
        }

        walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
