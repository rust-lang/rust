use rustc::lint::*;
use rustc::hir;
use rustc::hir::intravisit::{walk_expr, walk_fn, FnKind, NestedVisitorMap, Visitor};
use std::collections::HashMap;
use syntax::ast;
use syntax::codemap::Span;
use syntax::symbol::LocalInternedString;
use utils::{in_macro, span_lint};

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
declare_clippy_lint! {
    pub UNUSED_LABEL,
    complexity,
    "unused labels"
}

pub struct UnusedLabel;

struct UnusedLabelVisitor<'a, 'tcx: 'a> {
    labels: HashMap<LocalInternedString, Span>,
    cx: &'a LateContext<'a, 'tcx>,
}

impl LintPass for UnusedLabel {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_LABEL)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnusedLabel {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx hir::FnDecl,
        body: &'tcx hir::Body,
        span: Span,
        fn_id: ast::NodeId,
    ) {
        if in_macro(span) {
            return;
        }

        let mut v = UnusedLabelVisitor {
            cx,
            labels: HashMap::new(),
        };
        walk_fn(&mut v, kind, decl, body.id(), span, fn_id);

        for (label, span) in v.labels {
            span_lint(cx, UNUSED_LABEL, span, &format!("unused label `{}`", label));
        }
    }
}

impl<'a, 'tcx: 'a> Visitor<'tcx> for UnusedLabelVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprBreak(destination, _) | hir::ExprAgain(destination) => if let Some(label) = destination.label {
                self.labels.remove(&label.name.as_str());
            },
            hir::ExprLoop(_, Some(label), _) | hir::ExprWhile(_, _, Some(label)) => {
                self.labels.insert(label.name.as_str(), expr.span);
            },
            _ => (),
        }

        walk_expr(self, expr);
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.cx.tcx.hir)
    }
}
