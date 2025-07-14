use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{peel_blocks, peel_hir_expr_while, sym};
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks if both `.write(true)` and `.append(true)` methods are called
    /// on a same `OpenOptions`.
    ///
    /// ### Why is this bad?
    /// `.append(true)` already enables `write(true)`, making this one
    /// superfluous.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::fs::OpenOptions;
    /// let _ = OpenOptions::new()
    ///            .write(true)
    ///            .append(true)
    ///            .create(true)
    ///            .open("file.json");
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::fs::OpenOptions;
    /// let _ = OpenOptions::new()
    ///            .append(true)
    ///            .create(true)
    ///            .open("file.json");
    /// ```
    #[clippy::version = "1.76.0"]
    pub INEFFECTIVE_OPEN_OPTIONS,
    suspicious,
    "usage of both `write(true)` and `append(true)` on same `OpenOptions`"
}

declare_lint_pass!(IneffectiveOpenOptions => [INEFFECTIVE_OPEN_OPTIONS]);

impl<'tcx> LateLintPass<'tcx> for IneffectiveOpenOptions {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::MethodCall(name, recv, [_], _) = expr.kind
            && name.ident.name == sym::open
            && !expr.span.from_expansion()
            && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv).peel_refs(), sym::FsOpenOptions)
        {
            let mut append = false;
            let mut write = None;
            peel_hir_expr_while(recv, |e| {
                if let ExprKind::MethodCall(name, recv, args, call_span) = e.kind
                    && !e.span.from_expansion()
                {
                    if let [arg] = args
                        && let ExprKind::Lit(lit) = peel_blocks(arg).kind
                        && matches!(lit.node, LitKind::Bool(true))
                        && !arg.span.from_expansion()
                        && !lit.span.from_expansion()
                    {
                        match name.ident.name {
                            sym::append => append = true,
                            sym::write
                                if let Some(range) = call_span.map_range(cx, |_, text, range| {
                                    if text.get(..range.start)?.ends_with('.') {
                                        Some(range.start - 1..range.end)
                                    } else {
                                        None
                                    }
                                }) =>
                            {
                                write = Some(call_span.with_lo(range.start));
                            },
                            _ => {},
                        }
                    }
                    Some(recv)
                } else {
                    None
                }
            });

            if append && let Some(write_span) = write {
                span_lint_and_sugg(
                    cx,
                    INEFFECTIVE_OPEN_OPTIONS,
                    write_span,
                    "unnecessary use of `.write(true)` because there is `.append(true)`",
                    "remove `.write(true)`",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}
