use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{SpanRangeExt, snippet_with_applicability};
use rustc_ast::ast::{Expr, ExprKind, Mutability, UnOp};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::{BytePos, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `*&` and `*&mut` in expressions.
    ///
    /// ### Why is this bad?
    /// Immediately dereferencing a reference is no-op and
    /// makes the code less clear.
    ///
    /// ### Known problems
    /// Multiple dereference/addrof pairs are not handled so
    /// the suggested fix for `x = **&&y` is `x = *&y`, which is still incorrect.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let a = f(*&mut b);
    /// let c = *&d;
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// let a = f(b);
    /// let c = d;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DEREF_ADDROF,
    complexity,
    "use of `*&` or `*&mut` in an expression"
}

declare_lint_pass!(DerefAddrOf => [DEREF_ADDROF]);

fn without_parens(mut e: &Expr) -> &Expr {
    while let ExprKind::Paren(ref child_e) = e.kind {
        e = child_e;
    }
    e
}

impl EarlyLintPass for DerefAddrOf {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &Expr) {
        if let ExprKind::Unary(UnOp::Deref, ref deref_target) = e.kind
            && let ExprKind::AddrOf(_, ref mutability, ref addrof_target) = without_parens(deref_target).kind
            // NOTE(tesuji): `*&` forces rustc to const-promote the array to `.rodata` section.
            // See #12854 for details.
            && !matches!(addrof_target.kind, ExprKind::Array(_))
            && deref_target.span.eq_ctxt(e.span)
            && !addrof_target.span.from_expansion()
        {
            let mut applicability = Applicability::MachineApplicable;
            let sugg = if e.span.from_expansion() {
                if let Some(macro_source) = e.span.get_source_text(cx) {
                    // Remove leading whitespace from the given span
                    // e.g: ` $visitor` turns into `$visitor`
                    let trim_leading_whitespaces = |span: Span| {
                        span.get_source_text(cx)
                            .and_then(|snip| {
                                #[expect(clippy::cast_possible_truncation)]
                                snip.find(|c: char| !c.is_whitespace())
                                    .map(|pos| span.lo() + BytePos(pos as u32))
                            })
                            .map_or(span, |start_no_whitespace| e.span.with_lo(start_no_whitespace))
                    };

                    let mut generate_snippet = |pattern: &str| {
                        #[expect(clippy::cast_possible_truncation)]
                        macro_source.rfind(pattern).map(|pattern_pos| {
                            let rpos = pattern_pos + pattern.len();
                            let span_after_ref = e.span.with_lo(BytePos(e.span.lo().0 + rpos as u32));
                            let span = trim_leading_whitespaces(span_after_ref);
                            snippet_with_applicability(cx, span, "_", &mut applicability)
                        })
                    };

                    if *mutability == Mutability::Mut {
                        generate_snippet("mut")
                    } else {
                        generate_snippet("&")
                    }
                } else {
                    Some(snippet_with_applicability(cx, e.span, "_", &mut applicability))
                }
            } else {
                Some(snippet_with_applicability(
                    cx,
                    addrof_target.span,
                    "_",
                    &mut applicability,
                ))
            };
            if let Some(sugg) = sugg {
                span_lint_and_sugg(
                    cx,
                    DEREF_ADDROF,
                    e.span,
                    "immediately dereferencing a reference",
                    "try",
                    sugg.to_string(),
                    applicability,
                );
            }
        }
    }
}
