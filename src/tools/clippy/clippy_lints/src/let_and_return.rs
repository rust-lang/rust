use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{Block, Expr, ExprKind, PatKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::subst::GenericArgKind;
use rustc_session::{declare_lint_pass, declare_tool_lint};

use crate::utils::{in_macro, match_qpath, snippet_opt, span_lint_and_then};

declare_clippy_lint! {
    /// **What it does:** Checks for `let`-bindings, which are subsequently
    /// returned.
    ///
    /// **Why is this bad?** It is just extraneous code. Remove it to make your code
    /// more rusty.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn foo() -> String {
    ///     let x = String::new();
    ///     x
    /// }
    /// ```
    /// instead, use
    /// ```
    /// fn foo() -> String {
    ///     String::new()
    /// }
    /// ```
    pub LET_AND_RETURN,
    style,
    "creating a let-binding and then immediately returning it like `let x = expr; x` at the end of a block"
}

declare_lint_pass!(LetReturn => [LET_AND_RETURN]);

impl<'tcx> LateLintPass<'tcx> for LetReturn {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'_>) {
        // we need both a let-binding stmt and an expr
        if_chain! {
            if let Some(retexpr) = block.expr;
            if let Some(stmt) = block.stmts.iter().last();
            if let StmtKind::Local(local) = &stmt.kind;
            if local.ty.is_none();
            if local.attrs.is_empty();
            if let Some(initexpr) = &local.init;
            if let PatKind::Binding(.., ident, _) = local.pat.kind;
            if let ExprKind::Path(qpath) = &retexpr.kind;
            if match_qpath(qpath, &[&*ident.name.as_str()]);
            if !last_statement_borrows(cx, initexpr);
            if !in_external_macro(cx.sess(), initexpr.span);
            if !in_external_macro(cx.sess(), retexpr.span);
            if !in_external_macro(cx.sess(), local.span);
            if !in_macro(local.span);
            then {
                span_lint_and_then(
                    cx,
                    LET_AND_RETURN,
                    retexpr.span,
                    "returning the result of a `let` binding from a block",
                    |err| {
                        err.span_label(local.span, "unnecessary `let` binding");

                        if let Some(snippet) = snippet_opt(cx, initexpr.span) {
                            err.multipart_suggestion(
                                "return the expression directly",
                                vec![
                                    (local.span, String::new()),
                                    (retexpr.span, snippet),
                                ],
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.span_help(initexpr.span, "this expression can be directly returned");
                        }
                    },
                );
            }
        }
    }
}

fn last_statement_borrows<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    let mut visitor = BorrowVisitor { cx, borrows: false };
    walk_expr(&mut visitor, expr);
    visitor.borrows
}

struct BorrowVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    borrows: bool,
}

impl BorrowVisitor<'_, '_> {
    fn fn_def_id(&self, expr: &Expr<'_>) -> Option<DefId> {
        match &expr.kind {
            ExprKind::MethodCall(..) => self.cx.tables().type_dependent_def_id(expr.hir_id),
            ExprKind::Call(
                Expr {
                    kind: ExprKind::Path(qpath),
                    ..
                },
                ..,
            ) => self.cx.qpath_res(qpath, expr.hir_id).opt_def_id(),
            _ => None,
        }
    }
}

impl<'tcx> Visitor<'tcx> for BorrowVisitor<'_, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if self.borrows {
            return;
        }

        if let Some(def_id) = self.fn_def_id(expr) {
            self.borrows = self
                .cx
                .tcx
                .fn_sig(def_id)
                .output()
                .skip_binder()
                .walk()
                .any(|arg| matches!(arg.unpack(), GenericArgKind::Lifetime(_)));
        }

        walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
