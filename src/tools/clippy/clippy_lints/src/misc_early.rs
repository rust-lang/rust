use crate::utils::{constants, snippet_opt, span_lint, span_lint_and_help, span_lint_and_sugg, span_lint_and_then};
use rustc_ast::ast::{
    BindingMode, Expr, ExprKind, GenericParamKind, Generics, Lit, LitFloatType, LitIntType, LitKind, Mutability,
    NodeId, Pat, PatKind, UnOp,
};
use rustc_ast::visit::FnKind;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for structure field patterns bound to wildcards.
    ///
    /// **Why is this bad?** Using `..` instead is shorter and leaves the focus on
    /// the fields that are actually bound.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # struct Foo {
    /// #     a: i32,
    /// #     b: i32,
    /// #     c: i32,
    /// # }
    /// let f = Foo { a: 0, b: 0, c: 0 };
    ///
    /// // Bad
    /// match f {
    ///     Foo { a: _, b: 0, .. } => {},
    ///     Foo { a: _, b: _, c: _ } => {},
    /// }
    ///
    /// // Good
    /// match f {
    ///     Foo { b: 0, .. } => {},
    ///     Foo { .. } => {},
    /// }
    /// ```
    pub UNNEEDED_FIELD_PATTERN,
    restriction,
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
    /// // Bad
    /// fn foo(a: i32, _a: i32) {}
    ///
    /// // Good
    /// fn bar(a: i32, _b: i32) {}
    /// ```
    pub DUPLICATE_UNDERSCORE_ARGUMENT,
    style,
    "function arguments having names which only differ by an underscore"
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
    /// // Bad
    /// let y = 0x1a9BAcD;
    ///
    /// // Good
    /// let y = 0x1A9BACD;
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
    /// // Bad
    /// let y = 123832i32;
    ///
    /// // Good
    /// let y = 123832_i32;
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

declare_clippy_lint! {
    /// **What it does:** Checks for patterns in the form `name @ _`.
    ///
    /// **Why is this bad?** It's almost always more readable to just use direct
    /// bindings.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let v = Some("abc");
    ///
    /// // Bad
    /// match v {
    ///     Some(x) => (),
    ///     y @ _ => (),
    /// }
    ///
    /// // Good
    /// match v {
    ///     Some(x) => (),
    ///     y => (),
    /// }
    /// ```
    pub REDUNDANT_PATTERN,
    style,
    "using `name @ _` in a pattern"
}

declare_clippy_lint! {
    /// **What it does:** Checks for tuple patterns with a wildcard
    /// pattern (`_`) is next to a rest pattern (`..`).
    ///
    /// _NOTE_: While `_, ..` means there is at least one element left, `..`
    /// means there are 0 or more elements left. This can make a difference
    /// when refactoring, but shouldn't result in errors in the refactored code,
    /// since the wildcard pattern isn't used anyway.
    /// **Why is this bad?** The wildcard pattern is unneeded as the rest pattern
    /// can match that element as well.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # struct TupleStruct(u32, u32, u32);
    /// # let t = TupleStruct(1, 2, 3);
    /// // Bad
    /// match t {
    ///     TupleStruct(0, .., _) => (),
    ///     _ => (),
    /// }
    ///
    /// // Good
    /// match t {
    ///     TupleStruct(0, ..) => (),
    ///     _ => (),
    /// }
    /// ```
    pub UNNEEDED_WILDCARD_PATTERN,
    complexity,
    "tuple patterns with a wildcard pattern (`_`) is next to a rest pattern (`..`)"
}

declare_lint_pass!(MiscEarlyLints => [
    UNNEEDED_FIELD_PATTERN,
    DUPLICATE_UNDERSCORE_ARGUMENT,
    DOUBLE_NEG,
    MIXED_CASE_HEX_LITERALS,
    UNSEPARATED_LITERAL_SUFFIX,
    ZERO_PREFIXED_LITERAL,
    BUILTIN_TYPE_SHADOW,
    REDUNDANT_PATTERN,
    UNNEEDED_WILDCARD_PATTERN,
]);

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
                        &format!("this generic shadows the built-in type `{}`", name),
                    );
                }
            }
        }
    }

    fn check_pat(&mut self, cx: &EarlyContext<'_>, pat: &Pat) {
        if let PatKind::Struct(ref npat, ref pfields, _) = pat.kind {
            let mut wilds = 0;
            let type_name = npat
                .segments
                .last()
                .expect("A path must have at least one segment")
                .ident
                .name;

            for field in pfields {
                if let PatKind::Wild = field.pat.kind {
                    wilds += 1;
                }
            }
            if !pfields.is_empty() && wilds == pfields.len() {
                span_lint_and_help(
                    cx,
                    UNNEEDED_FIELD_PATTERN,
                    pat.span,
                    "all the struct fields are matched to a wildcard pattern, consider using `..`",
                    None,
                    &format!("try with `{} {{ .. }}` instead", type_name),
                );
                return;
            }
            if wilds > 0 {
                for field in pfields {
                    if let PatKind::Wild = field.pat.kind {
                        wilds -= 1;
                        if wilds > 0 {
                            span_lint(
                                cx,
                                UNNEEDED_FIELD_PATTERN,
                                field.span,
                                "you matched a field with a wildcard pattern, consider using `..` instead",
                            );
                        } else {
                            let mut normal = vec![];

                            for field in pfields {
                                match field.pat.kind {
                                    PatKind::Wild => {},
                                    _ => {
                                        if let Ok(n) = cx.sess().source_map().span_to_snippet(field.span) {
                                            normal.push(n);
                                        }
                                    },
                                }
                            }

                            span_lint_and_help(
                                cx,
                                UNNEEDED_FIELD_PATTERN,
                                field.span,
                                "you matched a field with a wildcard pattern, consider using `..` \
                                 instead",
                                None,
                                &format!("try with `{} {{ {}, .. }}`", type_name, normal[..].join(", ")),
                            );
                        }
                    }
                }
            }
        }

        if let PatKind::Ident(left, ident, Some(ref right)) = pat.kind {
            let left_binding = match left {
                BindingMode::ByRef(Mutability::Mut) => "ref mut ",
                BindingMode::ByRef(Mutability::Not) => "ref ",
                BindingMode::ByValue(..) => "",
            };

            if let PatKind::Wild = right.kind {
                span_lint_and_sugg(
                    cx,
                    REDUNDANT_PATTERN,
                    pat.span,
                    &format!(
                        "the `{} @ _` pattern can be written as just `{}`",
                        ident.name, ident.name,
                    ),
                    "try",
                    format!("{}{}", left_binding, ident.name),
                    Applicability::MachineApplicable,
                );
            }
        }

        check_unneeded_wildcard_pattern(cx, pat);
    }

    fn check_fn(&mut self, cx: &EarlyContext<'_>, fn_kind: FnKind<'_>, _: Span, _: NodeId) {
        let mut registered_names: FxHashMap<String, Span> = FxHashMap::default();

        for arg in &fn_kind.decl().inputs {
            if let PatKind::Ident(_, ident, None) = arg.pat.kind {
                let arg_name = ident.to_string();

                if let Some(arg_name) = arg_name.strip_prefix('_') {
                    if let Some(correspondence) = registered_names.get(arg_name) {
                        span_lint(
                            cx,
                            DUPLICATE_UNDERSCORE_ARGUMENT,
                            *correspondence,
                            &format!(
                                "`{}` already exists, having another argument having almost the same \
                                 name makes code comprehension and documentation more difficult",
                                arg_name
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
        match expr.kind {
            ExprKind::Unary(UnOp::Neg, ref inner) => {
                if let ExprKind::Unary(UnOp::Neg, _) = inner.kind {
                    span_lint(
                        cx,
                        DOUBLE_NEG,
                        expr.span,
                        "`--x` could be misinterpreted as pre-decrement by C programmers, is usually a no-op",
                    );
                }
            },
            ExprKind::Lit(ref lit) => Self::check_lit(cx, lit),
            _ => (),
        }
    }
}

impl MiscEarlyLints {
    fn check_lit(cx: &EarlyContext<'_>, lit: &Lit) {
        // We test if first character in snippet is a number, because the snippet could be an expansion
        // from a built-in macro like `line!()` or a proc-macro like `#[wasm_bindgen]`.
        // Note that this check also covers special case that `line!()` is eagerly expanded by compiler.
        // See <https://github.com/rust-lang/rust-clippy/issues/4507> for a regression.
        // FIXME: Find a better way to detect those cases.
        let lit_snip = match snippet_opt(cx, lit.span) {
            Some(snip) if snip.chars().next().map_or(false, |c| c.is_digit(10)) => snip,
            _ => return,
        };

        if let LitKind::Int(value, lit_int_type) = lit.kind {
            let suffix = match lit_int_type {
                LitIntType::Signed(ty) => ty.name_str(),
                LitIntType::Unsigned(ty) => ty.name_str(),
                LitIntType::Unsuffixed => "",
            };

            let maybe_last_sep_idx = if let Some(val) = lit_snip.len().checked_sub(suffix.len() + 1) {
                val
            } else {
                return; // It's useless so shouldn't lint.
            };
            // Do not lint when literal is unsuffixed.
            if !suffix.is_empty() && lit_snip.as_bytes()[maybe_last_sep_idx] != b'_' {
                span_lint_and_sugg(
                    cx,
                    UNSEPARATED_LITERAL_SUFFIX,
                    lit.span,
                    "integer type suffix should be separated by an underscore",
                    "add an underscore",
                    format!("{}_{}", &lit_snip[..=maybe_last_sep_idx], suffix),
                    Applicability::MachineApplicable,
                );
            }

            if lit_snip.starts_with("0x") {
                if maybe_last_sep_idx <= 2 {
                    // It's meaningless or causes range error.
                    return;
                }
                let mut seen = (false, false);
                for ch in lit_snip.as_bytes()[2..=maybe_last_sep_idx].iter() {
                    match ch {
                        b'a'..=b'f' => seen.0 = true,
                        b'A'..=b'F' => seen.1 = true,
                        _ => {},
                    }
                    if seen.0 && seen.1 {
                        span_lint(
                            cx,
                            MIXED_CASE_HEX_LITERALS,
                            lit.span,
                            "inconsistent casing in hexadecimal literal",
                        );
                        break;
                    }
                }
            } else if lit_snip.starts_with("0b") || lit_snip.starts_with("0o") {
                /* nothing to do */
            } else if value != 0 && lit_snip.starts_with('0') {
                span_lint_and_then(
                    cx,
                    ZERO_PREFIXED_LITERAL,
                    lit.span,
                    "this is a decimal constant",
                    |diag| {
                        diag.span_suggestion(
                            lit.span,
                            "if you mean to use a decimal constant, remove the `0` to avoid confusion",
                            lit_snip.trim_start_matches(|c| c == '_' || c == '0').to_string(),
                            Applicability::MaybeIncorrect,
                        );
                        diag.span_suggestion(
                            lit.span,
                            "if you mean to use an octal constant, use `0o`",
                            format!("0o{}", lit_snip.trim_start_matches(|c| c == '_' || c == '0')),
                            Applicability::MaybeIncorrect,
                        );
                    },
                );
            }
        } else if let LitKind::Float(_, LitFloatType::Suffixed(float_ty)) = lit.kind {
            let suffix = float_ty.name_str();
            let maybe_last_sep_idx = if let Some(val) = lit_snip.len().checked_sub(suffix.len() + 1) {
                val
            } else {
                return; // It's useless so shouldn't lint.
            };
            if lit_snip.as_bytes()[maybe_last_sep_idx] != b'_' {
                span_lint_and_sugg(
                    cx,
                    UNSEPARATED_LITERAL_SUFFIX,
                    lit.span,
                    "float type suffix should be separated by an underscore",
                    "add an underscore",
                    format!("{}_{}", &lit_snip[..=maybe_last_sep_idx], suffix),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

fn check_unneeded_wildcard_pattern(cx: &EarlyContext<'_>, pat: &Pat) {
    if let PatKind::TupleStruct(_, ref patterns) | PatKind::Tuple(ref patterns) = pat.kind {
        fn span_lint(cx: &EarlyContext<'_>, span: Span, only_one: bool) {
            span_lint_and_sugg(
                cx,
                UNNEEDED_WILDCARD_PATTERN,
                span,
                if only_one {
                    "this pattern is unneeded as the `..` pattern can match that element"
                } else {
                    "these patterns are unneeded as the `..` pattern can match those elements"
                },
                if only_one { "remove it" } else { "remove them" },
                "".to_string(),
                Applicability::MachineApplicable,
            );
        }

        if let Some(rest_index) = patterns.iter().position(|pat| pat.is_rest()) {
            if let Some((left_index, left_pat)) = patterns[..rest_index]
                .iter()
                .rev()
                .take_while(|pat| matches!(pat.kind, PatKind::Wild))
                .enumerate()
                .last()
            {
                span_lint(cx, left_pat.span.until(patterns[rest_index].span), left_index == 0);
            }

            if let Some((right_index, right_pat)) = patterns[rest_index + 1..]
                .iter()
                .take_while(|pat| matches!(pat.kind, PatKind::Wild))
                .enumerate()
                .last()
            {
                span_lint(
                    cx,
                    patterns[rest_index].span.shrink_to_hi().to(right_pat.span),
                    right_index == 0,
                );
            }
        }
    }
}
