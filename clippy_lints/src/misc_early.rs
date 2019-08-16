use crate::utils::{constants, snippet, snippet_opt, span_help_and_lint, span_lint, span_lint_and_then};
use if_chain::if_chain;
use rustc::lint::{in_external_macro, EarlyContext, EarlyLintPass, LintArray, LintContext, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use std::char;
use syntax::ast::*;
use syntax::source_map::Span;
use syntax::visit::{walk_expr, FnKind, Visitor};

declare_clippy_lint! {
    /// **What it does:** Checks for structure field patterns bound to wildcards.
    ///
    /// **Why is this bad?** Using `..` instead is shorter and leaves the focus on
    /// the fields that are actually bound.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// let { a: _, b: ref b, c: _ } = ..
    /// ```
    pub UNNEEDED_FIELD_PATTERN,
    style,
    "struct fields bound to a wildcard instead of using `..`"
}

declare_clippy_lint! {
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
    pub DUPLICATE_UNDERSCORE_ARGUMENT,
    style,
    "function arguments having names which only differ by an underscore"
}

declare_clippy_lint! {
    /// **What it does:** Detects closures called in the same expression where they
    /// are defined.
    ///
    /// **Why is this bad?** It is unnecessarily adding to the expression's
    /// complexity.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// (|| 42)()
    /// ```
    pub REDUNDANT_CLOSURE_CALL,
    complexity,
    "throwaway closures called in the expression they are defined"
}

declare_clippy_lint! {
    /// **What it does:** Detects expressions of the form `--x`.
    ///
    /// **Why is this bad?** It can mislead C/C++ programmers to think `x` was
    /// decremented.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let mut x = 3;
    /// --x;
    /// ```
    pub DOUBLE_NEG,
    style,
    "`--x`, which is a double negation of `x` and not a pre-decrement as in C/C++"
}

declare_clippy_lint! {
    /// **What it does:** Warns on hexadecimal literals with mixed-case letter
    /// digits.
    ///
    /// **Why is this bad?** It looks confusing.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let y = 0x1a9BAcD;
    /// ```
    pub MIXED_CASE_HEX_LITERALS,
    style,
    "hex literals whose letter digits are not consistently upper- or lowercased"
}

declare_clippy_lint! {
    /// **What it does:** Warns if literal suffixes are not separated by an
    /// underscore.
    ///
    /// **Why is this bad?** It is much less readable.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let y = 123832i32;
    /// ```
    pub UNSEPARATED_LITERAL_SUFFIX,
    pedantic,
    "literals whose suffix is not separated by an underscore"
}

declare_clippy_lint! {
    /// **What it does:** Warns if an integral constant literal starts with `0`.
    ///
    /// **Why is this bad?** In some languages (including the infamous C language
    /// and most of its
    /// family), this marks an octal constant. In Rust however, this is a decimal
    /// constant. This could
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
    pub ZERO_PREFIXED_LITERAL,
    complexity,
    "integer literals starting with `0`"
}

declare_clippy_lint! {
    /// **What it does:** Warns if a generic shadows a built-in type.
    ///
    /// **Why is this bad?** This gives surprising type errors.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```ignore
    /// impl<u32> Foo<u32> {
    ///     fn impl_func(&self) -> u32 {
    ///         42
    ///     }
    /// }
    /// ```
    pub BUILTIN_TYPE_SHADOW,
    style,
    "shadowing a builtin type"
}

declare_lint_pass!(MiscEarlyLints => [
    UNNEEDED_FIELD_PATTERN,
    DUPLICATE_UNDERSCORE_ARGUMENT,
    REDUNDANT_CLOSURE_CALL,
    DOUBLE_NEG,
    MIXED_CASE_HEX_LITERALS,
    UNSEPARATED_LITERAL_SUFFIX,
    ZERO_PREFIXED_LITERAL,
    BUILTIN_TYPE_SHADOW
]);

// Used to find `return` statements or equivalents e.g., `?`
struct ReturnVisitor {
    found_return: bool,
}

impl ReturnVisitor {
    fn new() -> Self {
        Self { found_return: false }
    }
}

impl<'ast> Visitor<'ast> for ReturnVisitor {
    fn visit_expr(&mut self, ex: &'ast Expr) {
        if let ExprKind::Ret(_) = ex.node {
            self.found_return = true;
        } else if let ExprKind::Try(_) = ex.node {
            self.found_return = true;
        }

        walk_expr(self, ex)
    }
}

impl EarlyLintPass for MiscEarlyLints {
    fn check_generics(&mut self, cx: &EarlyContext<'_>, gen: &Generics) {
        for param in &gen.params {
            if let GenericParamKind::Type { .. } = param.kind {
                let name = param.ident.as_str();
                if constants::BUILTIN_TYPES.contains(&&*name) {
                    span_lint(
                        cx,
                        BUILTIN_TYPE_SHADOW,
                        param.ident.span,
                        &format!("This generic shadows the built-in type `{}`", name),
                    );
                }
            }
        }
    }

    fn check_pat(&mut self, cx: &EarlyContext<'_>, pat: &Pat) {
        if let PatKind::Struct(ref npat, ref pfields, _) = pat.node {
            let mut wilds = 0;
            let type_name = npat
                .segments
                .last()
                .expect("A path must have at least one segment")
                .ident
                .name;

            for field in pfields {
                if let PatKind::Wild = field.pat.node {
                    wilds += 1;
                }
            }
            if !pfields.is_empty() && wilds == pfields.len() {
                span_help_and_lint(
                    cx,
                    UNNEEDED_FIELD_PATTERN,
                    pat.span,
                    "All the struct fields are matched to a wildcard pattern, consider using `..`.",
                    &format!("Try with `{} {{ .. }}` instead", type_name),
                );
                return;
            }
            if wilds > 0 {
                let mut normal = vec![];

                for field in pfields {
                    match field.pat.node {
                        PatKind::Wild => {},
                        _ => {
                            if let Ok(n) = cx.sess().source_map().span_to_snippet(field.span) {
                                normal.push(n);
                            }
                        },
                    }
                }
                for field in pfields {
                    if let PatKind::Wild = field.pat.node {
                        wilds -= 1;
                        if wilds > 0 {
                            span_lint(
                                cx,
                                UNNEEDED_FIELD_PATTERN,
                                field.span,
                                "You matched a field with a wildcard pattern. Consider using `..` instead",
                            );
                        } else {
                            span_help_and_lint(
                                cx,
                                UNNEEDED_FIELD_PATTERN,
                                field.span,
                                "You matched a field with a wildcard pattern. Consider using `..` \
                                 instead",
                                &format!("Try with `{} {{ {}, .. }}`", type_name, normal[..].join(", ")),
                            );
                        }
                    }
                }
            }
        }
    }

    fn check_fn(&mut self, cx: &EarlyContext<'_>, _: FnKind<'_>, decl: &FnDecl, _: Span, _: NodeId) {
        let mut registered_names: FxHashMap<String, Span> = FxHashMap::default();

        for arg in &decl.inputs {
            if let PatKind::Ident(_, ident, None) = arg.pat.node {
                let arg_name = ident.to_string();

                if arg_name.starts_with('_') {
                    if let Some(correspondence) = registered_names.get(&arg_name[1..]) {
                        span_lint(
                            cx,
                            DUPLICATE_UNDERSCORE_ARGUMENT,
                            *correspondence,
                            &format!(
                                "`{}` already exists, having another argument having almost the same \
                                 name makes code comprehension and documentation more difficult",
                                arg_name[1..].to_owned()
                            ),
                        );
                    }
                } else {
                    registered_names.insert(arg_name, arg.pat.span);
                }
            }
        }
    }

    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }
        match expr.node {
            ExprKind::Call(ref paren, _) => {
                if let ExprKind::Paren(ref closure) = paren.node {
                    if let ExprKind::Closure(_, _, _, ref decl, ref block, _) = closure.node {
                        let mut visitor = ReturnVisitor::new();
                        visitor.visit_expr(block);
                        if !visitor.found_return {
                            span_lint_and_then(
                                cx,
                                REDUNDANT_CLOSURE_CALL,
                                expr.span,
                                "Try not to call a closure in the expression where it is declared.",
                                |db| {
                                    if decl.inputs.is_empty() {
                                        let hint = snippet(cx, block.span, "..").into_owned();
                                        db.span_suggestion(
                                            expr.span,
                                            "Try doing something like: ",
                                            hint,
                                            Applicability::MachineApplicable, // snippet
                                        );
                                    }
                                },
                            );
                        }
                    }
                }
            },
            ExprKind::Unary(UnOp::Neg, ref inner) => {
                if let ExprKind::Unary(UnOp::Neg, _) = inner.node {
                    span_lint(
                        cx,
                        DOUBLE_NEG,
                        expr.span,
                        "`--x` could be misinterpreted as pre-decrement by C programmers, is usually a no-op",
                    );
                }
            },
            ExprKind::Lit(ref lit) => self.check_lit(cx, lit),
            _ => (),
        }
    }

    fn check_block(&mut self, cx: &EarlyContext<'_>, block: &Block) {
        for w in block.stmts.windows(2) {
            if_chain! {
                if let StmtKind::Local(ref local) = w[0].node;
                if let Option::Some(ref t) = local.init;
                if let ExprKind::Closure(..) = t.node;
                if let PatKind::Ident(_, ident, _) = local.pat.node;
                if let StmtKind::Semi(ref second) = w[1].node;
                if let ExprKind::Assign(_, ref call) = second.node;
                if let ExprKind::Call(ref closure, _) = call.node;
                if let ExprKind::Path(_, ref path) = closure.node;
                then {
                    if ident == path.segments[0].ident {
                        span_lint(
                            cx,
                            REDUNDANT_CLOSURE_CALL,
                            second.span,
                            "Closure called just once immediately after it was declared",
                        );
                    }
                }
            }
        }
    }
}

impl MiscEarlyLints {
    fn check_lit(self, cx: &EarlyContext<'_>, lit: &Lit) {
        if_chain! {
            if let LitKind::Int(value, ..) = lit.node;
            if let Some(src) = snippet_opt(cx, lit.span);
            if let Some(firstch) = src.chars().next();
            if char::to_digit(firstch, 10).is_some();
            then {
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
                            'a' ..= 'f' => seen.0 = true,
                            'A' ..= 'F' => seen.1 = true,
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
                            "if you mean to use a decimal constant, remove the `0` to remove confusion",
                            src.trim_start_matches(|c| c == '_' || c == '0').to_string(),
                            Applicability::MaybeIncorrect,
                        );
                        db.span_suggestion(
                            lit.span,
                            "if you mean to use an octal constant, use `0o`",
                            format!("0o{}", src.trim_start_matches(|c| c == '_' || c == '0')),
                            Applicability::MaybeIncorrect,
                        );
                    });
                }
            }
        }
        if_chain! {
            if let LitKind::Float(..) = lit.node;
            if let Some(src) = snippet_opt(cx, lit.span);
            if let Some(firstch) = src.chars().next();
            if char::to_digit(firstch, 10).is_some();
            then {
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
            }
        }
    }
}
