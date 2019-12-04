use crate::utils::span_lint;
use rustc::declare_lint_pass;
use rustc::hir;
use rustc::hir::intravisit::{walk_expr, walk_fn, FnKind, NestedVisitorMap, Visitor};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc_data_structures::fx::FxHashMap;
use rustc_session::declare_tool_lint;
use syntax::source_map::Span;
use syntax::symbol::Symbol;

declare_clippy_lint! {
    /// **What it does:** Checks for unused labels.
    ///
    /// **Why is this bad?** Maybe the label should be used in which case there is
    /// an error in the code or it should be removed.
    ///
    /// **Known problems:** Hopefully none.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// fn unused_label() {
    ///     'label: for i in 1..2 {
    ///         if i > 4 { continue }
    ///     }
    /// ```
    pub UNUSED_LABEL,
    complexity,
    "unused labels"
}

struct UnusedLabelVisitor<'a, 'tcx> {
    labels: FxHashMap<Symbol, Span>,
    cx: &'a LateContext<'a, 'tcx>,
}

declare_lint_pass!(UnusedLabel => [UNUSED_LABEL]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnusedLabel {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx hir::FnDecl,
        body: &'tcx hir::Body,
        span: Span,
        fn_id: hir::HirId,
    ) {
        if span.from_expansion() {
            return;
        }

        let mut v = UnusedLabelVisitor {
            cx,
            labels: FxHashMap::default(),
        };
        walk_fn(&mut v, kind, decl, body.id(), span, fn_id);

        for (label, span) in v.labels {
            span_lint(cx, UNUSED_LABEL, span, &format!("unused label `{}`", label));
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for UnusedLabelVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.kind {
            hir::ExprKind::Break(destination, _) | hir::ExprKind::Continue(destination) => {
                if let Some(label) = destination.label {
                    self.labels.remove(&label.ident.name);
                }
            },
            hir::ExprKind::Loop(_, Some(label), _) => {
                self.labels.insert(label.ident.name, expr.span);
            },
            _ => (),
        }

        walk_expr(self, expr);
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.cx.tcx.hir())
    }
}
