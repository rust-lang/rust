//use rustc_front::hir::*;

use rustc::lint::*;

use syntax::ast::*;

use utils::span_lint;

declare_lint!(pub UNNEEDED_BINDING, Warn,
              "Type fields are bound when not necessary");

#[derive(Copy, Clone)]
pub struct MiscEarly;

impl LintPass for MiscEarly {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNNEEDED_BINDING)
    }
}

impl EarlyLintPass for MiscEarly {
    fn check_pat(&mut self, cx: &EarlyContext, pat: &Pat) {
        if let PatStruct(_, ref pfields, _) = pat.node {
            let mut wilds = 0;

            for field in pfields {
                if field.node.pat.node == PatWild {
                    wilds += 1;
                }
            }
            if !pfields.is_empty() && wilds == pfields.len() {
                span_lint(cx, UNNEEDED_BINDING, pat.span,
                          "All the struct fields are matched to a wildcard pattern, \
                           consider using `..`.");
                return;
            }
            if wilds > 0 {
                for field in pfields {
                    if field.node.pat.node == PatWild {
                        span_lint(cx, UNNEEDED_BINDING, field.span,
                                  "You matched a field with a wildcard pattern. \
                                   Consider using `..` instead");
                    }
                }
            }
        }
    }
}
