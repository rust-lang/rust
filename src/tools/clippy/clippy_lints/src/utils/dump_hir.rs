use clippy_utils::get_attr;
use hir::TraitItem;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// It formats the attached node with `{:#?}` and writes the result to the
    /// standard output. This is intended for debugging.
    ///
    /// ### Examples
    /// ```rs
    /// #[clippy::dump]
    /// use std::mem;
    ///
    /// #[clippy::dump]
    /// fn foo(input: u32) -> u64 {
    ///     input as u64
    /// }
    /// ```
    pub DUMP_HIR,
    internal_warn,
    "helper to dump info about code"
}

declare_lint_pass!(DumpHir => [DUMP_HIR]);

impl<'tcx> LateLintPass<'tcx> for DumpHir {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        if has_attr(cx, item.hir_id()) {
            println!("{item:#?}");
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if has_attr(cx, expr.hir_id) {
            println!("{expr:#?}");
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx hir::Stmt<'_>) {
        match stmt.kind {
            hir::StmtKind::Expr(e) | hir::StmtKind::Semi(e) if has_attr(cx, e.hir_id) => return,
            _ => {},
        }
        if has_attr(cx, stmt.hir_id) {
            println!("{stmt:#?}");
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_>, item: &TraitItem<'_>) {
        if has_attr(cx, item.hir_id()) {
            println!("{item:#?}");
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_>, item: &hir::ImplItem<'_>) {
        if has_attr(cx, item.hir_id()) {
            println!("{item:#?}");
        }
    }
}

fn has_attr(cx: &LateContext<'_>, hir_id: hir::HirId) -> bool {
    let attrs = cx.tcx.hir().attrs(hir_id);
    get_attr(cx.sess(), attrs, "dump").count() > 0
}
