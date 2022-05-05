mod builtin_type_shadow;
mod double_neg;
mod literal_suffix;
mod mixed_case_hex_literals;
mod redundant_pattern;
mod unneeded_field_pattern;
mod unneeded_wildcard_pattern;
mod zero_prefixed_literal;

use clippy_utils::diagnostics::span_lint;
use clippy_utils::source::snippet_opt;
use rustc_ast::ast::{Expr, ExprKind, Generics, Lit, LitFloatType, LitIntType, LitKind, NodeId, Pat, PatKind};
use rustc_ast::visit::FnKind;
use rustc_data_structures::fx::FxHashMap;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for structure field patterns bound to wildcards.
    ///
    /// ### Why is this bad?
    /// Using `..` instead is shorter and leaves the focus on
    /// the fields that are actually bound.
    ///
    /// ### Example
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
    #[clippy::version = "pre 1.29.0"]
    pub UNNEEDED_FIELD_PATTERN,
    restriction,
    "struct fields bound to a wildcard instead of using `..`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for function arguments having the similar names
    /// differing by an underscore.
    ///
    /// ### Why is this bad?
    /// It affects code readability.
    ///
    /// ### Example
    /// ```rust
    /// // Bad
    /// fn foo(a: i32, _a: i32) {}
    ///
    /// // Good
    /// fn bar(a: i32, _b: i32) {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DUPLICATE_UNDERSCORE_ARGUMENT,
    style,
    "function arguments having names which only differ by an underscore"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects expressions of the form `--x`.
    ///
    /// ### Why is this bad?
    /// It can mislead C/C++ programmers to think `x` was
    /// decremented.
    ///
    /// ### Example
    /// ```rust
    /// let mut x = 3;
    /// --x;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DOUBLE_NEG,
    style,
    "`--x`, which is a double negation of `x` and not a pre-decrement as in C/C++"
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
    /// ```rust
    /// // Bad
    /// let y = 0x1a9BAcD;
    ///
    /// // Good
    /// let y = 0x1A9BACD;
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
    /// ### Why is this bad?
    /// Suffix style should be consistent.
    ///
    /// ### Example
    /// ```rust
    /// // Bad
    /// let y = 123832i32;
    ///
    /// // Good
    /// let y = 123832_i32;
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
    /// ### Why is this bad?
    /// Suffix style should be consistent.
    ///
    /// ### Example
    /// ```rust
    /// // Bad
    /// let y = 123832_i32;
    ///
    /// // Good
    /// let y = 123832i32;
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
    /// ### Why is this bad?
    /// The wildcard pattern is unneeded as the rest pattern
    /// can match that element as well.
    ///
    /// ### Example
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
    #[clippy::version = "1.40.0"]
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
    SEPARATED_LITERAL_SUFFIX,
    ZERO_PREFIXED_LITERAL,
    BUILTIN_TYPE_SHADOW,
    REDUNDANT_PATTERN,
    UNNEEDED_WILDCARD_PATTERN,
]);

impl EarlyLintPass for MiscEarlyLints {
    fn check_generics(&mut self, cx: &EarlyContext<'_>, gen: &Generics) {
        for param in &gen.params {
            builtin_type_shadow::check(cx, param);
        }
    }

    fn check_pat(&mut self, cx: &EarlyContext<'_>, pat: &Pat) {
        unneeded_field_pattern::check(cx, pat);
        redundant_pattern::check(cx, pat);
        unneeded_wildcard_pattern::check(cx, pat);
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

        if let ExprKind::Lit(ref lit) = expr.kind {
            MiscEarlyLints::check_lit(cx, lit);
        }
        double_neg::check(cx, expr);
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
            literal_suffix::check(cx, lit, &lit_snip, suffix, "integer");
            if lit_snip.starts_with("0x") {
                mixed_case_hex_literals::check(cx, lit, suffix, &lit_snip);
            } else if lit_snip.starts_with("0b") || lit_snip.starts_with("0o") {
                // nothing to do
            } else if value != 0 && lit_snip.starts_with('0') {
                zero_prefixed_literal::check(cx, lit, &lit_snip);
            }
        } else if let LitKind::Float(_, LitFloatType::Suffixed(float_ty)) = lit.kind {
            let suffix = float_ty.name_str();
            literal_suffix::check(cx, lit, &lit_snip, suffix, "float");
        }
    }
}
