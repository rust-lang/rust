use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use clippy_utils::{is_trait_method, is_try, match_trait_method, paths};
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unused written/read amount.
    ///
    /// ### Why is this bad?
    /// `io::Write::write(_vectored)` and
    /// `io::Read::read(_vectored)` are not guaranteed to
    /// process the entire buffer. They return how many bytes were processed, which
    /// might be smaller
    /// than a given buffer's length. If you don't need to deal with
    /// partial-write/read, use
    /// `write_all`/`read_exact` instead.
    ///
    /// When working with asynchronous code (either with the `futures`
    /// crate or with `tokio`), a similar issue exists for
    /// `AsyncWriteExt::write()` and `AsyncReadExt::read()` : these
    /// functions are also not guaranteed to process the entire
    /// buffer.  Your code should either handle partial-writes/reads, or
    /// call the `write_all`/`read_exact` methods on those traits instead.
    ///
    /// ### Known problems
    /// Detects only common patterns.
    ///
    /// ### Examples
    /// ```rust,ignore
    /// use std::io;
    /// fn foo<W: io::Write>(w: &mut W) -> io::Result<()> {
    ///     // must be `w.write_all(b"foo")?;`
    ///     w.write(b"foo")?;
    ///     Ok(())
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub UNUSED_IO_AMOUNT,
    correctness,
    "unused written/read amount"
}

declare_lint_pass!(UnusedIoAmount => [UNUSED_IO_AMOUNT]);

impl<'tcx> LateLintPass<'tcx> for UnusedIoAmount {
    fn check_stmt(&mut self, cx: &LateContext<'_>, s: &hir::Stmt<'_>) {
        let (hir::StmtKind::Semi(expr) | hir::StmtKind::Expr(expr)) = s.kind else {
            return;
        };

        match expr.kind {
            hir::ExprKind::Match(res, _, _) if is_try(cx, expr).is_some() => {
                if let hir::ExprKind::Call(func, [ref arg_0, ..]) = res.kind {
                    if matches!(
                        func.kind,
                        hir::ExprKind::Path(hir::QPath::LangItem(hir::LangItem::TryTraitBranch, ..))
                    ) {
                        check_map_error(cx, arg_0, expr);
                    }
                } else {
                    check_map_error(cx, res, expr);
                }
            },
            hir::ExprKind::MethodCall(path, arg_0, ..) => match path.ident.as_str() {
                "expect" | "unwrap" | "unwrap_or" | "unwrap_or_else" | "is_ok" | "is_err" => {
                    check_map_error(cx, arg_0, expr);
                },
                _ => (),
            },
            _ => (),
        }
    }
}

/// If `expr` is an (e).await, return the inner expression "e" that's being
/// waited on.  Otherwise return None.
fn try_remove_await<'a>(expr: &'a hir::Expr<'a>) -> Option<&hir::Expr<'a>> {
    if let hir::ExprKind::Match(expr, _, hir::MatchSource::AwaitDesugar) = expr.kind {
        if let hir::ExprKind::Call(func, [ref arg_0, ..]) = expr.kind {
            if matches!(
                func.kind,
                hir::ExprKind::Path(hir::QPath::LangItem(hir::LangItem::IntoFutureIntoFuture, ..))
            ) {
                return Some(arg_0);
            }
        }
    }

    None
}

fn check_map_error(cx: &LateContext<'_>, call: &hir::Expr<'_>, expr: &hir::Expr<'_>) {
    let mut call = call;
    while let hir::ExprKind::MethodCall(path, receiver, ..) = call.kind {
        if matches!(path.ident.as_str(), "or" | "or_else" | "ok") {
            call = receiver;
        } else {
            break;
        }
    }

    if let Some(call) = try_remove_await(call) {
        check_method_call(cx, call, expr, true);
    } else {
        check_method_call(cx, call, expr, false);
    }
}

fn check_method_call(cx: &LateContext<'_>, call: &hir::Expr<'_>, expr: &hir::Expr<'_>, is_await: bool) {
    if let hir::ExprKind::MethodCall(path, ..) = call.kind {
        let symbol = path.ident.as_str();
        let read_trait = if is_await {
            match_trait_method(cx, call, &paths::FUTURES_IO_ASYNCREADEXT)
                || match_trait_method(cx, call, &paths::TOKIO_IO_ASYNCREADEXT)
        } else {
            is_trait_method(cx, call, sym::IoRead)
        };
        let write_trait = if is_await {
            match_trait_method(cx, call, &paths::FUTURES_IO_ASYNCWRITEEXT)
                || match_trait_method(cx, call, &paths::TOKIO_IO_ASYNCWRITEEXT)
        } else {
            is_trait_method(cx, call, sym::IoWrite)
        };

        match (read_trait, write_trait, symbol, is_await) {
            (true, _, "read", false) => span_lint_and_help(
                cx,
                UNUSED_IO_AMOUNT,
                expr.span,
                "read amount is not handled",
                None,
                "use `Read::read_exact` instead, or handle partial reads",
            ),
            (true, _, "read", true) => span_lint_and_help(
                cx,
                UNUSED_IO_AMOUNT,
                expr.span,
                "read amount is not handled",
                None,
                "use `AsyncReadExt::read_exact` instead, or handle partial reads",
            ),
            (true, _, "read_vectored", _) => {
                span_lint(cx, UNUSED_IO_AMOUNT, expr.span, "read amount is not handled");
            },
            (_, true, "write", false) => span_lint_and_help(
                cx,
                UNUSED_IO_AMOUNT,
                expr.span,
                "written amount is not handled",
                None,
                "use `Write::write_all` instead, or handle partial writes",
            ),
            (_, true, "write", true) => span_lint_and_help(
                cx,
                UNUSED_IO_AMOUNT,
                expr.span,
                "written amount is not handled",
                None,
                "use `AsyncWriteExt::write_all` instead, or handle partial writes",
            ),
            (_, true, "write_vectored", _) => {
                span_lint(cx, UNUSED_IO_AMOUNT, expr.span, "written amount is not handled");
            },
            _ => (),
        }
    }
}
