use rustc::lint::*;
use std::collections::HashMap;
use std::char;
use syntax::ast::*;
use syntax::codemap::Span;
use syntax::visit::FnKind;
use utils::{constants, span_lint, span_help_and_lint, snippet, snippet_opt, span_lint_and_then, in_external_macro};

/// **What it does:** Checks for structure field patterns bound to wildcards.
///
/// **Why is this bad?** Using `..` instead is shorter and leaves the focus on
/// the fields that are actually bound.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let { a: _, b: ref b, c: _ } = ..
/// ```
declare_lint! {
    pub UNNEEDED_FIELD_PATTERN,
    Warn,
    "struct fields bound to a wildcard instead of using `..`"
}

/// **What it does:** Checks for function arguments having the similar names
/// differing by an underscore.
///
/// **Why is this bad?** It affects code readability.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// fn foo(a: i32, _a: i32) {}
/// ```
declare_lint! {
    pub DUPLICATE_UNDERSCORE_ARGUMENT,
    Warn,
    "function arguments having names which only differ by an underscore"
}

/// **What it does:** Detects closures called in the same expression where they are defined.
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
    pub REDUNDANT_CLOSURE_CALL,
    Warn,
    "throwaway closures called in the expression they are defined"
}

/// **What it does:** Detects expressions of the form `--x`.
///
/// **Why is this bad?** It can mislead C/C++ programmers to think `x` was
/// decremented.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// --x;
/// ```
declare_lint! {
    pub DOUBLE_NEG,
    Warn,
    "`--x`, which is a double negation of `x` and not a pre-decrement as in C/C++"
}

/// **What it does:** Warns on hexadecimal literals with mixed-case letter digits.
///
/// **Why is this bad?** It looks confusing.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let y = 0x1a9BAcD;
/// ```
declare_lint! {
    pub MIXED_CASE_HEX_LITERALS,
    Warn,
    "hex literals whose letter digits are not consistently upper- or lowercased"
}

/// **What it does:** Warns if literal suffixes are not separated by an underscore.
///
/// **Why is this bad?** It is much less readable.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let y = 123832i32;
/// ```
declare_lint! {
    pub UNSEPARATED_LITERAL_SUFFIX,
    Allow,
    "literals whose suffix is not separated by an underscore"
}

/// **What it does:** Warns if an integral constant literal starts with `0`.
///
/// **Why is this bad?** In some languages (including the infamous C language and most of its
/// family), this marks an octal constant. In Rust however, this is a decimal constant. This could
/// be confusing for both the writer and a reader of the constant.
///
/// **Known problems:** None.
///
/// **Example:**
///
/// In Rust:
/// ```rust
/// fn main() {
///     let a = 0123;
///     println!("{}", a);
/// }
/// ```
///
/// prints `123`, while in C:
///
/// ```c
/// #include <stdio.h>
///
/// int main() {
///     int a = 0123;
///     printf("%d\n", a);
/// }
/// ```
///
/// prints `83` (as `83 == 0o123` while `123 == 0o173`).
declare_lint! {
    pub ZERO_PREFIXED_LITERAL,
    Warn,
    "integer literals starting with `0`"
}

/// **What it does:** Warns if a generic shadows a built-in type.
///
/// **Why is this bad?** This gives surprising type errors.
///
/// **Known problems:** None.
///
/// **Example:**
///
/// ```rust
/// impl<u32> Foo<u32> {
///     fn impl_func(&self) -> u32 {
///         42
///     }
/// }
/// ```
declare_lint! {
    pub BUILTIN_TYPE_SHADOW,
    Warn,
    "shadowing a builtin type"
}

#[derive(Copy, Clone)]
pub struct MiscEarly;

impl LintPass for MiscEarly {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNNEEDED_FIELD_PATTERN,
                    DUPLICATE_UNDERSCORE_ARGUMENT,
                    REDUNDANT_CLOSURE_CALL,
                    DOUBLE_NEG,
                    MIXED_CASE_HEX_LITERALS,
                    UNSEPARATED_LITERAL_SUFFIX,
                    ZERO_PREFIXED_LITERAL,
                    BUILTIN_TYPE_SHADOW)
    }
}

impl EarlyLintPass for MiscEarly {
    fn check_generics(&mut self, cx: &EarlyContext, gen: &Generics) {
        for ty in &gen.ty_params {
            let name = ty.ident.name.as_str();
            if constants::BUILTIN_TYPES.contains(&&*name) {
                span_lint(cx,
                          BUILTIN_TYPE_SHADOW,
                          ty.span,
                          &format!("This generic shadows the built-in type `{}`", name));
            }
        }
    }

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

    fn check_fn(&mut self, cx: &EarlyContext, _: FnKind, decl: &FnDecl, _: Span, _: NodeId) {
        let mut registered_names: HashMap<String, Span> = HashMap::new();

        for arg in &decl.inputs {
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
        if in_external_macro(cx, expr.span) {
            return;
        }
        match expr.node {
            ExprKind::Call(ref paren, _) => {
                if let ExprKind::Paren(ref closure) = paren.node {
                    if let ExprKind::Closure(_, ref decl, ref block, _) = closure.node {
                        span_lint_and_then(cx,
                                           REDUNDANT_CLOSURE_CALL,
                                           expr.span,
                                           "Try not to call a closure in the expression where it is declared.",
                                           |db| if decl.inputs.is_empty() {
                                               let hint = snippet(cx, block.span, "..").into_owned();
                                               db.span_suggestion(expr.span, "Try doing something like: ", hint);
                                           });
                    }
                }
            },
            ExprKind::Unary(UnOp::Neg, ref inner) => {
                if let ExprKind::Unary(UnOp::Neg, _) = inner.node {
                    span_lint(cx,
                              DOUBLE_NEG,
                              expr.span,
                              "`--x` could be misinterpreted as pre-decrement by C programmers, is usually a no-op");
                }
            },
            ExprKind::Lit(ref lit) => self.check_lit(cx, lit),
            _ => (),
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
                    span_lint(
                        cx,
                        REDUNDANT_CLOSURE_CALL,
                        second.span,
                        "Closure called just once immediately after it was declared",
                    );
                }
            }}
        }
    }
}

impl MiscEarly {
    fn check_lit(&self, cx: &EarlyContext, lit: &Lit) {
        if_let_chain! {[
            let LitKind::Int(value, ..) = lit.node,
            let Some(src) = snippet_opt(cx, lit.span),
            let Some(firstch) = src.chars().next(),
            char::to_digit(firstch, 10).is_some()
        ], {
            let mut prev = '\0';
            for ch in src.chars() {
                if ch == 'i' || ch == 'u' {
                    if prev != '_' {
                        span_lint(cx, UNSEPARATED_LITERAL_SUFFIX, lit.span,
                                    "integer type suffix should be separated by an underscore");
                    }
                    break;
                }
                prev = ch;
            }
            if src.starts_with("0x") {
                let mut seen = (false, false);
                for ch in src.chars() {
                    match ch {
                        'a' ... 'f' => seen.0 = true,
                        'A' ... 'F' => seen.1 = true,
                        'i' | 'u'   => break,   // start of suffix already
                        _ => ()
                    }
                }
                if seen.0 && seen.1 {
                    span_lint(cx, MIXED_CASE_HEX_LITERALS, lit.span,
                                "inconsistent casing in hexadecimal literal");
                }
            } else if src.starts_with("0b") || src.starts_with("0o") {
                /* nothing to do */
            } else if value != 0 && src.starts_with('0') {
                span_lint_and_then(cx,
                                    ZERO_PREFIXED_LITERAL,
                                    lit.span,
                                    "this is a decimal constant",
                                    |db| {
                    db.span_suggestion(
                        lit.span,
                        "if you mean to use a decimal constant, remove the `0` to remove confusion:",
                        src.trim_left_matches('0').to_string(),
                    );
                    db.span_suggestion(
                        lit.span,
                        "if you mean to use an octal constant, use `0o`:",
                        format!("0o{}", src.trim_left_matches('0')),
                    );
                });
            }
        }}
        if_let_chain! {[
            let LitKind::Float(..) = lit.node,
            let Some(src) = snippet_opt(cx, lit.span),
            let Some(firstch) = src.chars().next(),
            char::to_digit(firstch, 10).is_some()
        ], {
            let mut prev = '\0';
            for ch in src.chars() {
                if ch == 'f' {
                    if prev != '_' {
                        span_lint(cx, UNSEPARATED_LITERAL_SUFFIX, lit.span,
                                    "float type suffix should be separated by an underscore");
                    }
                    break;
                }
                prev = ch;
            }
        }}
    }
}
