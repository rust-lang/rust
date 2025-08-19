use std::str::Utf8Error;

use rustc_ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::source_map::Spanned;
use rustc_span::sym;

use crate::lints::InvalidFromUtf8Diag;
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `invalid_from_utf8_unchecked` lint checks for calls to
    /// `std::str::from_utf8_unchecked` and `std::str::from_utf8_unchecked_mut`
    /// with a known invalid UTF-8 value.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// # #[allow(unused)]
    /// unsafe {
    ///     std::str::from_utf8_unchecked(b"Ru\x82st");
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Creating such a `str` would result in undefined behavior as per documentation
    /// for `std::str::from_utf8_unchecked` and `std::str::from_utf8_unchecked_mut`.
    pub INVALID_FROM_UTF8_UNCHECKED,
    Deny,
    "using a non UTF-8 literal in `std::str::from_utf8_unchecked`"
}

declare_lint! {
    /// The `invalid_from_utf8` lint checks for calls to
    /// `std::str::from_utf8` and `std::str::from_utf8_mut`
    /// with a known invalid UTF-8 value.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #[allow(unused)]
    /// std::str::from_utf8(b"Ru\x82st");
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Trying to create such a `str` would always return an error as per documentation
    /// for `std::str::from_utf8` and `std::str::from_utf8_mut`.
    pub INVALID_FROM_UTF8,
    Warn,
    "using a non UTF-8 literal in `std::str::from_utf8`"
}

declare_lint_pass!(InvalidFromUtf8 => [INVALID_FROM_UTF8_UNCHECKED, INVALID_FROM_UTF8]);

impl<'tcx> LateLintPass<'tcx> for InvalidFromUtf8 {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Call(path, [arg]) = expr.kind
            && let ExprKind::Path(ref qpath) = path.kind
            && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
            && let Some(diag_item) = cx.tcx.get_diagnostic_name(def_id)
            && [
                sym::str_from_utf8,
                sym::str_from_utf8_mut,
                sym::str_from_utf8_unchecked,
                sym::str_from_utf8_unchecked_mut,
                sym::str_inherent_from_utf8,
                sym::str_inherent_from_utf8_mut,
                sym::str_inherent_from_utf8_unchecked,
                sym::str_inherent_from_utf8_unchecked_mut,
            ]
            .contains(&diag_item)
        {
            let lint = |label, utf8_error: Utf8Error| {
                let method = diag_item.as_str().strip_prefix("str_").unwrap();
                let method = if let Some(method) = method.strip_prefix("inherent_") {
                    format!("str::{method}")
                } else {
                    format!("std::str::{method}")
                };
                let valid_up_to = utf8_error.valid_up_to();
                let is_unchecked_variant = diag_item.as_str().contains("unchecked");

                cx.emit_span_lint(
                    if is_unchecked_variant {
                        INVALID_FROM_UTF8_UNCHECKED
                    } else {
                        INVALID_FROM_UTF8
                    },
                    expr.span,
                    if is_unchecked_variant {
                        InvalidFromUtf8Diag::Unchecked { method, valid_up_to, label }
                    } else {
                        InvalidFromUtf8Diag::Checked { method, valid_up_to, label }
                    },
                )
            };

            let mut init = cx.expr_or_init_with_outside_body(arg);
            while let ExprKind::AddrOf(.., inner) = init.kind {
                init = cx.expr_or_init_with_outside_body(inner);
            }
            match init.kind {
                ExprKind::Lit(Spanned { node: lit, .. }) => {
                    if let LitKind::ByteStr(byte_sym, _) = &lit
                        && let Err(utf8_error) = std::str::from_utf8(byte_sym.as_byte_str())
                    {
                        lint(init.span, utf8_error);
                    }
                }
                ExprKind::Array(args) => {
                    let elements = args
                        .iter()
                        .map(|e| match &e.kind {
                            ExprKind::Lit(Spanned { node: lit, .. }) => match lit {
                                LitKind::Byte(b) => Some(*b),
                                LitKind::Int(b, _) => Some(b.get() as u8),
                                _ => None,
                            },
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>();

                    if let Some(elements) = elements
                        && let Err(utf8_error) = std::str::from_utf8(&elements)
                    {
                        lint(init.span, utf8_error);
                    }
                }
                _ => {}
            }
        }
    }
}
