use rustc::lint::*;

use std::collections::HashMap;

use syntax::ast::*;
use syntax::codemap::Span;
use syntax::visit::FnKind;

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

/// **What it does:** This lint `Warn`s on function arguments having the similar names differing by an underscore
///
/// **Why is this bad?** It affects code readability
///
/// **Known problems:** None.
///
/// **Example:** `fn foo(a: i32, _a: i32) {}`
declare_lint!(pub DUPLICATE_UNDERSCORE_ARGUMENT, Warn,
              "Function arguments having names which only differ by an underscore");

#[derive(Copy, Clone)]
pub struct MiscEarly;

impl LintPass for MiscEarly {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNNEEDED_FIELD_PATTERN, DUPLICATE_UNDERSCORE_ARGUMENT)
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
                span_help_and_lint(cx,
                                   UNNEEDED_FIELD_PATTERN,
                                   pat.span,
                                   "All the struct fields are matched to a wildcard pattern, consider using `..`.",
                                   &format!("Try with `{} {{ .. }}` instead", type_name));
                return;
            }
            if wilds > 0 {
                let mut normal = vec![];

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
                            span_lint(cx,
                                      UNNEEDED_FIELD_PATTERN,
                                      field.span,
                                      "You matched a field with a wildcard pattern. Consider using `..` instead");
                        } else {
                            span_help_and_lint(cx,
                                               UNNEEDED_FIELD_PATTERN,
                                               field.span,
                                               "You matched a field with a wildcard pattern. Consider using `..` \
                                                instead",
                                               &format!("Try with `{} {{ {}, .. }}`",
                                                        type_name,
                                                        normal[..].join(", ")));
                        }
                    }
                }
            }
        }
    }

    fn check_fn(&mut self, cx: &EarlyContext, _: FnKind, decl: &FnDecl, _: &Block, _: Span, _: NodeId) {
        let mut registered_names: HashMap<String, Span> = HashMap::new();

        for ref arg in &decl.inputs {
            if let PatIdent(_, sp_ident, None) = arg.pat.node {
                let arg_name = sp_ident.node.to_string();

                if arg_name.starts_with("_") {
                    if let Some(correspondance) = registered_names.get(&arg_name[1..]) {
                        span_lint(cx,
                                  DUPLICATE_UNDERSCORE_ARGUMENT,
                                  *correspondance,
                                  &format!("`{}` already exists, having another argument having almost the same \
                                            name makes code comprehension and documentation more difficult",
                                           arg_name[1..].to_owned()));
                    }
                } else {
                    registered_names.insert(arg_name, arg.pat.span.clone());
                }
            }
        }
    }
}
