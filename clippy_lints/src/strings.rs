use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;
use rustc_span::sym;

use if_chain::if_chain;

use crate::utils::SpanlessEq;
use crate::utils::{get_parent_expr, is_allowed, is_type_diagnostic_item, span_lint, span_lint_and_sugg};

declare_clippy_lint! {
    /// **What it does:** Checks for string appends of the form `x = x + y` (without
    /// `let`!).
    ///
    /// **Why is this bad?** It's not really bad, but some people think that the
    /// `.push_str(_)` method is more readable.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let mut x = "Hello".to_owned();
    /// x = x + ", World";
    ///
    /// // More readable
    /// x += ", World";
    /// x.push_str(", World");
    /// ```
    pub STRING_ADD_ASSIGN,
    pedantic,
    "using `x = x + ..` where x is a `String` instead of `push_str()`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for all instances of `x + _` where `x` is of type
    /// `String`, but only if [`string_add_assign`](#string_add_assign) does *not*
    /// match.
    ///
    /// **Why is this bad?** It's not bad in and of itself. However, this particular
    /// `Add` implementation is asymmetric (the other operand need not be `String`,
    /// but `x` does), while addition as mathematically defined is symmetric, also
    /// the `String::push_str(_)` function is a perfectly good replacement.
    /// Therefore, some dislike it and wish not to have it in their code.
    ///
    /// That said, other people think that string addition, having a long tradition
    /// in other languages is actually fine, which is why we decided to make this
    /// particular lint `allow` by default.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let x = "Hello".to_owned();
    /// x + ", World";
    /// ```
    pub STRING_ADD,
    restriction,
    "using `x + ..` where x is a `String` instead of `push_str()`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for the `as_bytes` method called on string literals
    /// that contain only ASCII characters.
    ///
    /// **Why is this bad?** Byte string literals (e.g., `b"foo"`) can be used
    /// instead. They are shorter but less discoverable than `as_bytes()`.
    ///
    /// **Known Problems:**
    /// `"str".as_bytes()` and the suggested replacement of `b"str"` are not
    /// equivalent because they have different types. The former is `&[u8]`
    /// while the latter is `&[u8; 3]`. That means in general they will have a
    /// different set of methods and different trait implementations.
    ///
    /// ```compile_fail
    /// fn f(v: Vec<u8>) {}
    ///
    /// f("...".as_bytes().to_owned()); // works
    /// f(b"...".to_owned()); // does not work, because arg is [u8; 3] not Vec<u8>
    ///
    /// fn g(r: impl std::io::Read) {}
    ///
    /// g("...".as_bytes()); // works
    /// g(b"..."); // does not work
    /// ```
    ///
    /// The actual equivalent of `"str".as_bytes()` with the same type is not
    /// `b"str"` but `&b"str"[..]`, which is a great deal of punctuation and not
    /// more readable than a function call.
    ///
    /// **Example:**
    /// ```rust
    /// // Bad
    /// let bs = "a byte string".as_bytes();
    ///
    /// // Good
    /// let bs = b"a byte string";
    /// ```
    pub STRING_LIT_AS_BYTES,
    nursery,
    "calling `as_bytes` on a string literal instead of using a byte string literal"
}

declare_lint_pass!(StringAdd => [STRING_ADD, STRING_ADD_ASSIGN]);

impl<'tcx> LateLintPass<'tcx> for StringAdd {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if in_external_macro(cx.sess(), e.span) {
            return;
        }

        if let ExprKind::Binary(
            Spanned {
                node: BinOpKind::Add, ..
            },
            ref left,
            _,
        ) = e.kind
        {
            if is_string(cx, left) {
                if !is_allowed(cx, STRING_ADD_ASSIGN, e.hir_id) {
                    let parent = get_parent_expr(cx, e);
                    if let Some(p) = parent {
                        if let ExprKind::Assign(ref target, _, _) = p.kind {
                            // avoid duplicate matches
                            if SpanlessEq::new(cx).eq_expr(target, left) {
                                return;
                            }
                        }
                    }
                }
                span_lint(
                    cx,
                    STRING_ADD,
                    e.span,
                    "you added something to a string. Consider using `String::push_str()` instead",
                );
            }
        } else if let ExprKind::Assign(ref target, ref src, _) = e.kind {
            if is_string(cx, target) && is_add(cx, src, target) {
                span_lint(
                    cx,
                    STRING_ADD_ASSIGN,
                    e.span,
                    "you assigned the result of adding something to this string. Consider using \
                     `String::push_str()` instead",
                );
            }
        }
    }
}

fn is_string(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(e).peel_refs(), sym::string_type)
}

fn is_add(cx: &LateContext<'_>, src: &Expr<'_>, target: &Expr<'_>) -> bool {
    match src.kind {
        ExprKind::Binary(
            Spanned {
                node: BinOpKind::Add, ..
            },
            ref left,
            _,
        ) => SpanlessEq::new(cx).eq_expr(target, left),
        ExprKind::Block(ref block, _) => {
            block.stmts.is_empty() && block.expr.as_ref().map_or(false, |expr| is_add(cx, expr, target))
        },
        _ => false,
    }
}

// Max length a b"foo" string can take
const MAX_LENGTH_BYTE_STRING_LIT: usize = 32;

declare_lint_pass!(StringLitAsBytes => [STRING_LIT_AS_BYTES]);

impl<'tcx> LateLintPass<'tcx> for StringLitAsBytes {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        use crate::utils::{snippet, snippet_with_applicability};
        use rustc_ast::LitKind;

        if_chain! {
            if let ExprKind::MethodCall(path, _, args, _) = &e.kind;
            if path.ident.name == sym!(as_bytes);
            if let ExprKind::Lit(lit) = &args[0].kind;
            if let LitKind::Str(lit_content, _) = &lit.node;
            then {
                let callsite = snippet(cx, args[0].span.source_callsite(), r#""foo""#);
                let mut applicability = Applicability::MachineApplicable;
                if callsite.starts_with("include_str!") {
                    span_lint_and_sugg(
                        cx,
                        STRING_LIT_AS_BYTES,
                        e.span,
                        "calling `as_bytes()` on `include_str!(..)`",
                        "consider using `include_bytes!(..)` instead",
                        snippet_with_applicability(cx, args[0].span, r#""foo""#, &mut applicability).replacen(
                            "include_str",
                            "include_bytes",
                            1,
                        ),
                        applicability,
                    );
                } else if lit_content.as_str().is_ascii()
                    && lit_content.as_str().len() <= MAX_LENGTH_BYTE_STRING_LIT
                    && !args[0].span.from_expansion()
                {
                    span_lint_and_sugg(
                        cx,
                        STRING_LIT_AS_BYTES,
                        e.span,
                        "calling `as_bytes()` on a string literal",
                        "consider using a byte string literal instead",
                        format!(
                            "b{}",
                            snippet_with_applicability(cx, args[0].span, r#""foo""#, &mut applicability)
                        ),
                        applicability,
                    );
                }
            }
        }
    }
}
