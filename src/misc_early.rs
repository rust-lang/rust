//use rustc_front::hir::*;

use rustc::lint::*;

use syntax::ast::*;

use utils::{span_lint, span_help_and_lint};

/// **What it does:** This lint `Warn`s on struct field patterns bound to wildcards.
///
/// **Why is this bad?** Using `..` instead is shorter and leaves the focus on the fields that are actually bound.
///
/// **Known problems:** None.
///
/// **Example:** `let { a: _, b: ref b, c: _ } = ..`
declare_lint!(pub UNNEEDED_FIELD_PATTERN, Warn,
              "Struct fields are bound to a wildcard instead of using `..`");

#[derive(Copy, Clone)]
pub struct MiscEarly;

impl LintPass for MiscEarly {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNNEEDED_FIELD_PATTERN)
    }
}

impl EarlyLintPass for MiscEarly {
    fn check_pat(&mut self, cx: &EarlyContext, pat: &Pat) {
        if let PatStruct(ref npat, ref pfields, _) = pat.node {
            let mut wilds = 0;
            let type_name = match npat.segments.last() {
                Some(elem) => format!("{}", elem.identifier.name),
                None => String::new(),
            };

            for field in pfields {
                if field.node.pat.node == PatWild {
                    wilds += 1;
                }
            }
            if !pfields.is_empty() && wilds == pfields.len() {
                span_help_and_lint(cx, UNNEEDED_FIELD_PATTERN, pat.span,
                                   "All the struct fields are matched to a wildcard pattern, \
                                    consider using `..`.",
                                   &format!("Try with `{} {{ .. }}` instead",
                                            type_name));
                return;
            }
            if wilds > 0 {
                let mut normal = vec!();

                for field in pfields {
                    if field.node.pat.node != PatWild {
                        if let Ok(n) = cx.sess().codemap().span_to_snippet(field.span) {
                            normal.push(n);
                        }
                    }
                }
                for field in pfields {
                    if field.node.pat.node == PatWild {
                        wilds -= 1;
                        if wilds > 0 {
                            span_lint(cx, UNNEEDED_FIELD_PATTERN, field.span,
                                      "You matched a field with a wildcard pattern. \
                                       Consider using `..` instead");
                        } else {
                            span_help_and_lint(cx, UNNEEDED_FIELD_PATTERN, field.span,
                                               "You matched a field with a wildcard pattern. \
                                                Consider using `..` instead",
                                               &format!("Try with `{} {{ {}, .. }}`",
                                                        type_name,
                                                        normal[..].join(", ")));
                        }
                    }
                }
            }
        }
    }
}
