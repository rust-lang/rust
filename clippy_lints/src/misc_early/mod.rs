mod builtin_type_shadow;
mod literal_suffix;
mod mixed_case_hex_literals;
mod redundant_at_rest_pattern;
mod redundant_pattern;
mod unneeded_field_pattern;
mod unneeded_wildcard_pattern;
mod zero_prefixed_literal;

use clippy_utils::source::snippet_opt;
use rustc_ast::ast::{Expr, ExprKind, Generics, LitFloatType, LitIntType, LitKind, Pat};
use rustc_ast::token;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for structure field patterns bound to wildcards.
    ///
    /// ### Why restrict this?
    /// Using `..` instead is shorter and leaves the focus on
    /// the fields that are actually bound.
    ///
    /// ### Example
    /// ```no_run
    /// # struct Foo {
    /// #     a: i32,
    /// #     b: i32,
    /// #     c: i32,
    /// # }
    /// let f = Foo { a: 0, b: 0, c: 0 };
    ///
    /// match f {
    ///     Foo { a: _, b: 0, .. } => {},
    ///     Foo { a: _, b: _, c: _ } => {},
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # struct Foo {
    /// #     a: i32,
    /// #     b: i32,
    /// #     c: i32,
    /// # }
    /// let f = Foo { a: 0, b: 0, c: 0 };
    ///
    /// match f {
    ///     Foo { b: 0, .. } => {},
    ///     Foo { .. } => {},
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub UNNEEDED_FIELD_PATTERN,
    restriction,
    "struct fields bound to a wildcard instead of using `..`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns on hexadecimal literals with mixed-case letter
    /// digits.
    ///
    /// ### Why is this bad?
    /// It looks confusing.
    ///
    /// ### Example
    /// ```no_run
    /// # let _ =
    /// 0x1a9BAcD
    /// # ;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let _ =
    /// 0x1A9BACD
    /// # ;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MIXED_CASE_HEX_LITERALS,
    style,
    "hex literals whose letter digits are not consistently upper- or lowercased"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns if literal suffixes are not separated by an
    /// underscore.
    /// To enforce unseparated literal suffix style,
    /// see the `separated_literal_suffix` lint.
    ///
    /// ### Why restrict this?
    /// Suffix style should be consistent.
    ///
    /// ### Example
    /// ```no_run
    /// # let _ =
    /// 123832i32
    /// # ;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let _ =
    /// 123832_i32
    /// # ;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub UNSEPARATED_LITERAL_SUFFIX,
    restriction,
    "literals whose suffix is not separated by an underscore"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns if literal suffixes are separated by an underscore.
    /// To enforce separated literal suffix style,
    /// see the `unseparated_literal_suffix` lint.
    ///
    /// ### Why restrict this?
    /// Suffix style should be consistent.
    ///
    /// ### Example
    /// ```no_run
    /// # let _ =
    /// 123832_i32
    /// # ;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let _ =
    /// 123832i32
    /// # ;
    /// ```
    #[clippy::version = "1.58.0"]
    pub SEPARATED_LITERAL_SUFFIX,
    restriction,
    "literals whose suffix is separated by an underscore"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns if an integral constant literal starts with `0`.
    ///
    /// ### Why is this bad?
    /// In some languages (including the infamous C language
    /// and most of its
    /// family), this marks an octal constant. In Rust however, this is a decimal
    /// constant. This could
    /// be confusing for both the writer and a reader of the constant.
    ///
    /// ### Example
    ///
    /// In Rust:
    /// ```no_run
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
    #[clippy::version = "pre 1.29.0"]
    pub ZERO_PREFIXED_LITERAL,
    complexity,
    "integer literals starting with `0`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns if a generic shadows a built-in type.
    ///
    /// ### Why is this bad?
    /// This gives surprising type errors.
    ///
    /// ### Example
    ///
    /// ```ignore
    /// impl<u32> Foo<u32> {
    ///     fn impl_func(&self) -> u32 {
    ///         42
    ///     }
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub BUILTIN_TYPE_SHADOW,
    style,
    "shadowing a builtin type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for patterns in the form `name @ _`.
    ///
    /// ### Why is this bad?
    /// It's almost always more readable to just use direct
    /// bindings.
    ///
    /// ### Example
    /// ```no_run
    /// # let v = Some("abc");
    /// match v {
    ///     Some(x) => (),
    ///     y @ _ => (),
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let v = Some("abc");
    /// match v {
    ///     Some(x) => (),
    ///     y => (),
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub REDUNDANT_PATTERN,
    style,
    "using `name @ _` in a pattern"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for tuple patterns with a wildcard
    /// pattern (`_`) is next to a rest pattern (`..`).
    ///
    /// _NOTE_: While `_, ..` means there is at least one element left, `..`
    /// means there are 0 or more elements left. This can make a difference
    /// when refactoring, but shouldn't result in errors in the refactored code,
    /// since the wildcard pattern isn't used anyway.
    ///
    /// ### Why is this bad?
    /// The wildcard pattern is unneeded as the rest pattern
    /// can match that element as well.
    ///
    /// ### Example
    /// ```no_run
    /// # struct TupleStruct(u32, u32, u32);
    /// # let t = TupleStruct(1, 2, 3);
    /// match t {
    ///     TupleStruct(0, .., _) => (),
    ///     _ => (),
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # struct TupleStruct(u32, u32, u32);
    /// # let t = TupleStruct(1, 2, 3);
    /// match t {
    ///     TupleStruct(0, ..) => (),
    ///     _ => (),
    /// }
    /// ```
    #[clippy::version = "1.40.0"]
    pub UNNEEDED_WILDCARD_PATTERN,
    complexity,
    "tuple patterns with a wildcard pattern (`_`) is next to a rest pattern (`..`)"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `[all @ ..]` patterns.
    ///
    /// ### Why is this bad?
    /// In all cases, `all` works fine and can often make code simpler, as you possibly won't need
    /// to convert from say a `Vec` to a slice by dereferencing.
    ///
    /// ### Example
    /// ```rust,ignore
    /// if let [all @ ..] = &*v {
    ///     // NOTE: Type is a slice here
    ///     println!("all elements: {all:#?}");
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// if let all = v {
    ///     // NOTE: Type is a `Vec` here
    ///     println!("all elements: {all:#?}");
    /// }
    /// // or
    /// println!("all elements: {v:#?}");
    /// ```
    #[clippy::version = "1.72.0"]
    pub REDUNDANT_AT_REST_PATTERN,
    complexity,
    "checks for `[all @ ..]` where `all` would suffice"
}

declare_lint_pass!(MiscEarlyLints => [
    UNNEEDED_FIELD_PATTERN,
    MIXED_CASE_HEX_LITERALS,
    UNSEPARATED_LITERAL_SUFFIX,
    SEPARATED_LITERAL_SUFFIX,
    ZERO_PREFIXED_LITERAL,
    BUILTIN_TYPE_SHADOW,
    REDUNDANT_PATTERN,
    UNNEEDED_WILDCARD_PATTERN,
    REDUNDANT_AT_REST_PATTERN,
]);

impl EarlyLintPass for MiscEarlyLints {
    fn check_generics(&mut self, cx: &EarlyContext<'_>, generics: &Generics) {
        for param in &generics.params {
            builtin_type_shadow::check(cx, param);
        }
    }

    fn check_pat(&mut self, cx: &EarlyContext<'_>, pat: &Pat) {
        if pat.span.in_external_macro(cx.sess().source_map()) {
            return;
        }

        unneeded_field_pattern::check(cx, pat);
        redundant_pattern::check(cx, pat);
        redundant_at_rest_pattern::check(cx, pat);
        unneeded_wildcard_pattern::check(cx, pat);
    }

    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if expr.span.in_external_macro(cx.sess().source_map()) {
            return;
        }

        if let ExprKind::Lit(lit) = expr.kind {
            MiscEarlyLints::check_lit(cx, lit, expr.span);
        }
    }
}

impl MiscEarlyLints {
    fn check_lit(cx: &EarlyContext<'_>, lit: token::Lit, span: Span) {
        // We test if first character in snippet is a number, because the snippet could be an expansion
        // from a built-in macro like `line!()` or a proc-macro like `#[wasm_bindgen]`.
        // Note that this check also covers special case that `line!()` is eagerly expanded by compiler.
        // See <https://github.com/rust-lang/rust-clippy/issues/4507> for a regression.
        // FIXME: Find a better way to detect those cases.
        let lit_snip = match snippet_opt(cx, span) {
            Some(snip) if snip.starts_with(|c: char| c.is_ascii_digit()) => snip,
            _ => return,
        };

        let lit_kind = LitKind::from_token_lit(lit);
        if let Ok(LitKind::Int(value, lit_int_type)) = lit_kind {
            let suffix = match lit_int_type {
                LitIntType::Signed(ty) => ty.name_str(),
                LitIntType::Unsigned(ty) => ty.name_str(),
                LitIntType::Unsuffixed => "",
            };
            literal_suffix::check(cx, span, &lit_snip, suffix, "integer");
            if lit_snip.starts_with("0x") {
                mixed_case_hex_literals::check(cx, span, suffix, &lit_snip);
            } else if lit_snip.starts_with("0b") || lit_snip.starts_with("0o") {
                // nothing to do
            } else if value != 0 && lit_snip.starts_with('0') {
                zero_prefixed_literal::check(cx, span, &lit_snip);
            }
        } else if let Ok(LitKind::Float(_, LitFloatType::Suffixed(float_ty))) = lit_kind {
            let suffix = float_ty.name_str();
            literal_suffix::check(cx, span, &lit_snip, suffix, "float");
        }
    }
}
