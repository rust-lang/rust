use rustc::lint::*;
use std::collections::HashMap;
use syntax::ast::*;
use syntax::codemap::Span;
use syntax::visit::FnKind;
use utils::{span_lint, span_help_and_lint, snippet, span_lint_and_then};
/// **What it does:** This lint checks for structure field patterns bound to wildcards.
///
/// **Why is this bad?** Using `..` instead is shorter and leaves the focus on the fields that are actually bound.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let { a: _, b: ref b, c: _ } = ..
/// ```
declare_lint! {
    pub UNNEEDED_FIELD_PATTERN, Warn,
    "Struct fields are bound to a wildcard instead of using `..`"
}

/// **What it does:** This lint checks for function arguments having the similar names differing by an underscore
///
/// **Why is this bad?** It affects code readability
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// fn foo(a: i32, _a: i32) {}
/// ```
declare_lint! {
    pub DUPLICATE_UNDERSCORE_ARGUMENT, Warn,
    "Function arguments having names which only differ by an underscore"
}

/// **What it does:** This lint detects closures called in the same expression where they are defined.
///
/// **Why is this bad?** It is unnecessarily adding to the expression's complexity.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// (|| 42)()
/// ```
declare_lint! {
    pub REDUNDANT_CLOSURE_CALL, Warn,
    "Closures should not be called in the expression they are defined"
}

/// **What it does:** This lint detects expressions of the form `--x`
///
/// **Why is this bad?** It can mislead C/C++ programmers to think `x` was decremented.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// --x;
/// ```
declare_lint! {
    pub DOUBLE_NEG, Warn,
    "`--x` is a double negation of `x` and not a pre-decrement as in C or C++"
}


#[derive(Copy, Clone)]
pub struct MiscEarly;

impl LintPass for MiscEarly {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNNEEDED_FIELD_PATTERN, DUPLICATE_UNDERSCORE_ARGUMENT, REDUNDANT_CLOSURE_CALL, DOUBLE_NEG)
    }
}

impl EarlyLintPass for MiscEarly {
    fn check_pat(&mut self, cx: &EarlyContext, pat: &Pat) {
        if let PatKind::Struct(ref npat, ref pfields, _) = pat.node {
            let mut wilds = 0;
            let type_name = npat.segments.last().expect("A path must have at least one segment").identifier.name;

            for field in pfields {
                if field.node.pat.node == PatKind::Wild {
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
                    if field.node.pat.node != PatKind::Wild {
                        if let Ok(n) = cx.sess().codemap().span_to_snippet(field.span) {
                            normal.push(n);
                        }
                    }
                }
                for field in pfields {
                    if field.node.pat.node == PatKind::Wild {
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
            if let PatKind::Ident(_, sp_ident, None) = arg.pat.node {
                let arg_name = sp_ident.node.to_string();

                if arg_name.starts_with('_') {
                    if let Some(correspondence) = registered_names.get(&arg_name[1..]) {
                        span_lint(cx,
                                  DUPLICATE_UNDERSCORE_ARGUMENT,
                                  *correspondence,
                                  &format!("`{}` already exists, having another argument having almost the same \
                                            name makes code comprehension and documentation more difficult",
                                           arg_name[1..].to_owned()));;
                    }
                } else {
                    registered_names.insert(arg_name, arg.pat.span);
                }
            }
        }
    }

    fn check_expr(&mut self, cx: &EarlyContext, expr: &Expr) {
        match expr.node {
            ExprKind::Call(ref paren, _) => {
                if let ExprKind::Paren(ref closure) = paren.node {
                    if let ExprKind::Closure(_, ref decl, ref block, _) = closure.node {
                        span_lint_and_then(cx,
                                           REDUNDANT_CLOSURE_CALL,
                                           expr.span,
                                           "Try not to call a closure in the expression where it is declared.",
                                           |db| {
                                               if decl.inputs.is_empty() {
                                                   let hint = snippet(cx, block.span, "..").into_owned();
                                                   db.span_suggestion(expr.span, "Try doing something like: ", hint);
                                               }
                                           });
                    }
                }
            }
            ExprKind::Unary(UnOp::Neg, ref inner) => {
                if let ExprKind::Unary(UnOp::Neg, _) = inner.node {
                    span_lint(cx,
                              DOUBLE_NEG,
                              expr.span,
                              "`--x` could be misinterpreted as pre-decrement by C programmers, is usually a no-op");
    }
            }
            _ => ()
        }
    }

    fn check_block(&mut self, cx: &EarlyContext, block: &Block) {
        for w in block.stmts.windows(2) {
            if_let_chain! {[
                let StmtKind::Local(ref local) = w[0].node,
                let Option::Some(ref t) = local.init,
                let ExprKind::Closure(_, _, _, _) = t.node,
                let PatKind::Ident(_, sp_ident, _) = local.pat.node,
                let StmtKind::Semi(ref second) = w[1].node,
                let ExprKind::Assign(_, ref call) = second.node,
                let ExprKind::Call(ref closure, _) = call.node,
                let ExprKind::Path(_, ref path) = closure.node
            ], {
                if sp_ident.node == (&path.segments[0]).identifier {
                    span_lint(cx, REDUNDANT_CLOSURE_CALL, second.span, "Closure called just once immediately after it was declared");
                }
            }}
        }
    }
}
