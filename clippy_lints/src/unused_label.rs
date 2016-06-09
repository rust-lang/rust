use rustc::lint::*;
use rustc::hir;
use rustc::hir::intravisit::{FnKind, Visitor, walk_expr, walk_fn};
use std::collections::HashMap;
use syntax::ast;
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use utils::{in_macro, span_lint};

/// **What it does:** This lint checks for unused labels.
///
/// **Why is this bad?** Maybe the label should be used in which case there is an error in the
/// code or it should be removed.
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
declare_lint! {
    pub UNUSED_LABEL,
    Warn,
    "unused label"
}

pub struct UnusedLabel;

#[derive(Default)]
struct UnusedLabelVisitor {
    labels: HashMap<InternedString, Span>,
}

impl UnusedLabelVisitor {
    pub fn new() -> UnusedLabelVisitor {
        ::std::default::Default::default()
    }
}

impl LintPass for UnusedLabel {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_LABEL)
    }
}

impl LateLintPass for UnusedLabel {
    fn check_fn(&mut self, cx: &LateContext, kind: FnKind, decl: &hir::FnDecl, body: &hir::Block, span: Span, _: ast::NodeId) {
        if in_macro(cx, span) {
            return;
        }

        let mut v = UnusedLabelVisitor::new();
        walk_fn(&mut v, kind, decl, body, span);

        for (label, span) in v.labels {
            span_lint(cx, UNUSED_LABEL, span, &format!("unused label `{}`", label));
        }
    }
}

impl<'v> Visitor<'v> for UnusedLabelVisitor {
    fn visit_expr(&mut self, expr: &hir::Expr) {
        match expr.node {
            hir::ExprBreak(Some(label)) |
            hir::ExprAgain(Some(label)) => {
                self.labels.remove(&label.node.as_str());
            }
            hir::ExprLoop(_, Some(label)) |
            hir::ExprWhile(_, _, Some(label)) => {
                self.labels.insert(label.node.as_str(), expr.span);
            }
            _ => (),
        }

        walk_expr(self, expr);
    }
}
