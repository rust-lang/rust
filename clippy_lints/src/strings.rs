use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::source_map::Spanned;

use crate::utils::SpanlessEq;
use crate::utils::{get_parent_expr, is_allowed, match_type, paths, span_lint, span_lint_and_sugg, walk_ptrs_ty};

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
    /// **Known Problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let bs = "a byte string".as_bytes();
    /// ```
    pub STRING_LIT_AS_BYTES,
    style,
    "calling `as_bytes` on a string literal instead of using a byte string literal"
}

declare_lint_pass!(StringAdd => [STRING_ADD, STRING_ADD_ASSIGN]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for StringAdd {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if let ExprKind::Binary(
            Spanned {
                node: BinOpKind::Add, ..
            },
            ref left,
            _,
        ) = e.node
        {
            if is_string(cx, left) {
                if !is_allowed(cx, STRING_ADD_ASSIGN, e.hir_id) {
                    let parent = get_parent_expr(cx, e);
                    if let Some(p) = parent {
                        if let ExprKind::Assign(ref target, _) = p.node {
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
        } else if let ExprKind::Assign(ref target, ref src) = e.node {
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

fn is_string(cx: &LateContext<'_, '_>, e: &Expr) -> bool {
    match_type(cx, walk_ptrs_ty(cx.tables.expr_ty(e)), &paths::STRING)
}

fn is_add(cx: &LateContext<'_, '_>, src: &Expr, target: &Expr) -> bool {
    match src.node {
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for StringLitAsBytes {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        use crate::utils::{in_macro_or_desugar, snippet, snippet_with_applicability};
        use syntax::ast::{LitKind, StrStyle};

        if let ExprKind::MethodCall(ref path, _, ref args) = e.node {
            if path.ident.name == sym!(as_bytes) {
                if let ExprKind::Lit(ref lit) = args[0].node {
                    if let LitKind::Str(ref lit_content, style) = lit.node {
                        let callsite = snippet(cx, args[0].span.source_callsite(), r#""foo""#);
                        let expanded = if let StrStyle::Raw(n) = style {
                            let term = (0..n).map(|_| '#').collect::<String>();
                            format!("r{0}\"{1}\"{0}", term, lit_content.as_str())
                        } else {
                            format!("\"{}\"", lit_content.as_str())
                        };
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
                        } else if callsite == expanded
                            && lit_content.as_str().chars().all(|c| c.is_ascii())
                            && lit_content.as_str().len() <= MAX_LENGTH_BYTE_STRING_LIT
                            && !in_macro_or_desugar(args[0].span)
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
    }
}
