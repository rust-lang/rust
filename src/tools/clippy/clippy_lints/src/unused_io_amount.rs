use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::macros::{is_panic, root_macro_call_first_node};
use clippy_utils::{is_res_lang_ctor, paths, peel_blocks, sym};
use hir::{ExprKind, HirId, PatKind};
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

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
    ///     w.write(b"foo")?;
    ///     Ok(())
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// use std::io;
    /// fn foo<W: io::Write>(w: &mut W) -> io::Result<()> {
    ///     w.write_all(b"foo")?;
    ///     Ok(())
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub UNUSED_IO_AMOUNT,
    correctness,
    "unused written/read amount"
}

declare_lint_pass!(UnusedIoAmount => [UNUSED_IO_AMOUNT]);

#[derive(Copy, Clone)]
enum IoOp {
    AsyncWrite(bool),
    AsyncRead(bool),
    SyncRead(bool),
    SyncWrite(bool),
}

impl<'tcx> LateLintPass<'tcx> for UnusedIoAmount {
    /// We perform the check on the block level.
    /// If we want to catch match and if expressions that act as returns of the block
    ///   we need to check them at `check_expr` or `check_block` as they are not stmts
    ///   but we can't check them at `check_expr` because we need the broader context
    ///   because we should do this only for the final expression of the block, and not for
    ///   `StmtKind::Let` which binds values => the io amount is used.
    ///
    /// To check for unused io amount in stmts, we only consider `StmtKind::Semi`.
    /// `StmtKind::Let` is not considered because it binds values => the io amount is used.
    /// `StmtKind::Expr` is not considered because requires unit type => the io amount is used.
    /// `StmtKind::Item` is not considered because it's not an expression.
    ///
    /// We then check the individual expressions via `check_expr`. We use the same logic for
    /// semi expressions and the final expression as we need to check match and if expressions
    /// for binding of the io amount to `Ok(_)`.
    ///
    /// We explicitly check for the match source to be Normal as it needs special logic
    /// to consider the arms, and we want to avoid breaking the logic for situations where things
    /// get desugared to match.
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx hir::Block<'tcx>) {
        let fn_def_id = block.hir_id.owner.to_def_id();
        if let Some(impl_id) = cx.tcx.impl_of_assoc(fn_def_id)
            && let Some(trait_id) = cx.tcx.trait_id_of_impl(impl_id)
        {
            // We don't want to lint inside io::Read or io::Write implementations, as the author has more
            // information about their trait implementation than our lint, see https://github.com/rust-lang/rust-clippy/issues/4836
            if let Some(trait_name) = cx.tcx.get_diagnostic_name(trait_id)
                && matches!(trait_name, sym::IoRead | sym::IoWrite)
            {
                return;
            }

            let async_paths = [
                &paths::TOKIO_IO_ASYNCREADEXT,
                &paths::TOKIO_IO_ASYNCWRITEEXT,
                &paths::FUTURES_IO_ASYNCREADEXT,
                &paths::FUTURES_IO_ASYNCWRITEEXT,
            ];

            if async_paths.into_iter().any(|path| path.matches(cx, trait_id)) {
                return;
            }
        }

        for stmt in block.stmts {
            if let hir::StmtKind::Semi(exp) = stmt.kind {
                check_expr(cx, exp);
            }
        }

        if let Some(exp) = block.expr
            && matches!(
                exp.kind,
                ExprKind::If(_, _, _) | ExprKind::Match(_, _, hir::MatchSource::Normal)
            )
        {
            check_expr(cx, exp);
        }
    }
}

fn non_consuming_err_arm<'a>(cx: &LateContext<'a>, arm: &hir::Arm<'a>) -> bool {
    // if there is a guard, we consider the result to be consumed
    if arm.guard.is_some() {
        return false;
    }
    if is_unreachable_or_panic(cx, arm.body) {
        // if the body is unreachable or there is a panic,
        // we consider the result to be consumed
        return false;
    }

    if let PatKind::TupleStruct(ref path, [inner_pat], _) = arm.pat.kind {
        return is_res_lang_ctor(cx, cx.qpath_res(path, inner_pat.hir_id), hir::LangItem::ResultErr);
    }

    false
}

fn non_consuming_ok_arm<'a>(cx: &LateContext<'a>, arm: &hir::Arm<'a>) -> bool {
    // if there is a guard, we consider the result to be consumed
    if arm.guard.is_some() {
        return false;
    }
    if is_unreachable_or_panic(cx, arm.body) {
        // if the body is unreachable or there is a panic,
        // we consider the result to be consumed
        return false;
    }

    if is_ok_wild_or_dotdot_pattern(cx, arm.pat) {
        return true;
    }
    false
}

fn check_expr<'a>(cx: &LateContext<'a>, expr: &'a hir::Expr<'a>) {
    match expr.kind {
        ExprKind::If(cond, _, _)
            if let ExprKind::Let(hir::LetExpr { pat, init, .. }) = cond.kind
                && is_ok_wild_or_dotdot_pattern(cx, pat)
                && let Some(op) = should_lint(cx, init) =>
        {
            emit_lint(cx, cond.span, cond.hir_id, op, &[pat.span]);
        },
        // we will capture only the case where the match is Ok( ) or Err( )
        // prefer to match the minimum possible, and expand later if needed
        // to avoid false positives on something as used as this
        ExprKind::Match(expr, [arm1, arm2], hir::MatchSource::Normal) if let Some(op) = should_lint(cx, expr) => {
            if non_consuming_ok_arm(cx, arm1) && non_consuming_err_arm(cx, arm2) {
                emit_lint(cx, expr.span, expr.hir_id, op, &[arm1.pat.span]);
            }
            if non_consuming_ok_arm(cx, arm2) && non_consuming_err_arm(cx, arm1) {
                emit_lint(cx, expr.span, expr.hir_id, op, &[arm2.pat.span]);
            }
        },
        ExprKind::Match(_, _, hir::MatchSource::Normal) => {},
        _ if let Some(op) = should_lint(cx, expr) => {
            emit_lint(cx, expr.span, expr.hir_id, op, &[]);
        },
        _ => {},
    }
}

fn should_lint<'a>(cx: &LateContext<'a>, mut inner: &'a hir::Expr<'a>) -> Option<IoOp> {
    inner = unpack_match(inner);
    inner = unpack_try(inner);
    inner = unpack_call_chain(inner);
    inner = unpack_await(inner);
    // we type-check it to get whether it's a read/write or their vectorized forms
    // and keep only the ones that are produce io amount
    check_io_mode(cx, inner)
}

fn is_ok_wild_or_dotdot_pattern<'a>(cx: &LateContext<'a>, pat: &hir::Pat<'a>) -> bool {
    // the if checks whether we are in a result Ok( ) pattern
    // and the return checks whether it is unhandled

    if let PatKind::TupleStruct(ref path, inner_pat, _) = pat.kind
        // we check against Result::Ok to avoid linting on Err(_) or something else.
        && is_res_lang_ctor(cx, cx.qpath_res(path, pat.hir_id), hir::LangItem::ResultOk)
    {
        if matches!(inner_pat, []) {
            return true;
        }

        if let [cons_pat] = inner_pat
            && matches!(cons_pat.kind, PatKind::Wild)
        {
            return true;
        }
        return false;
    }
    false
}

// this is partially taken from panic_unimplemented
fn is_unreachable_or_panic(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    let expr = peel_blocks(expr);
    let Some(macro_call) = root_macro_call_first_node(cx, expr) else {
        return false;
    };
    if is_panic(cx, macro_call.def_id) {
        return !cx.tcx.hir_is_inside_const_context(expr.hir_id);
    }
    cx.tcx.is_diagnostic_item(sym::unreachable_macro, macro_call.def_id)
}

fn unpack_call_chain<'a>(mut expr: &'a hir::Expr<'a>) -> &'a hir::Expr<'a> {
    while let ExprKind::MethodCall(path, receiver, ..) = expr.kind {
        if matches!(
            path.ident.name,
            sym::unwrap
                | sym::expect
                | sym::unwrap_or
                | sym::unwrap_or_else
                | sym::ok
                | sym::is_ok
                | sym::is_err
                | sym::or_else
                | sym::or
        ) {
            expr = receiver;
        } else {
            break;
        }
    }
    expr
}

fn unpack_try<'a>(mut expr: &'a hir::Expr<'a>) -> &'a hir::Expr<'a> {
    while let ExprKind::Call(func, [arg_0]) = expr.kind
        && matches!(
            func.kind,
            ExprKind::Path(hir::QPath::LangItem(hir::LangItem::TryTraitBranch, ..))
        )
    {
        expr = arg_0;
    }
    expr
}

fn unpack_match<'a>(mut expr: &'a hir::Expr<'a>) -> &'a hir::Expr<'a> {
    while let ExprKind::Match(res, _, _) = expr.kind {
        expr = res;
    }
    expr
}

/// If `expr` is an (e).await, return the inner expression "e" that's being
/// waited on.  Otherwise return None.
fn unpack_await<'a>(expr: &'a hir::Expr<'a>) -> &'a hir::Expr<'a> {
    if let ExprKind::Match(expr, _, hir::MatchSource::AwaitDesugar) = expr.kind
        && let ExprKind::Call(func, [arg_0]) = expr.kind
        && matches!(
            func.kind,
            ExprKind::Path(hir::QPath::LangItem(hir::LangItem::IntoFutureIntoFuture, ..))
        )
    {
        return arg_0;
    }
    expr
}

/// Check whether the current expr is a function call for an IO operation
fn check_io_mode(cx: &LateContext<'_>, call: &hir::Expr<'_>) -> Option<IoOp> {
    let ExprKind::MethodCall(path, ..) = call.kind else {
        return None;
    };

    let vectorized = match path.ident.as_str() {
        "write_vectored" | "read_vectored" => true,
        "write" | "read" => false,
        _ => {
            return None;
        },
    };

    if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(call.hir_id)
        && let Some(trait_def_id) = cx.tcx.trait_of_assoc(method_def_id)
    {
        if let Some(diag_name) = cx.tcx.get_diagnostic_name(trait_def_id) {
            match diag_name {
                sym::IoRead => Some(IoOp::SyncRead(vectorized)),
                sym::IoWrite => Some(IoOp::SyncWrite(vectorized)),
                _ => None,
            }
        } else if paths::FUTURES_IO_ASYNCREADEXT.matches(cx, trait_def_id)
            || paths::TOKIO_IO_ASYNCREADEXT.matches(cx, trait_def_id)
        {
            Some(IoOp::AsyncRead(vectorized))
        } else if paths::TOKIO_IO_ASYNCWRITEEXT.matches(cx, trait_def_id)
            || paths::FUTURES_IO_ASYNCWRITEEXT.matches(cx, trait_def_id)
        {
            Some(IoOp::AsyncWrite(vectorized))
        } else {
            None
        }
    } else {
        None
    }
}

fn emit_lint(cx: &LateContext<'_>, span: Span, at: HirId, op: IoOp, wild_cards: &[Span]) {
    let (msg, help) = match op {
        IoOp::AsyncRead(false) => (
            "read amount is not handled",
            Some("use `AsyncReadExt::read_exact` instead, or handle partial reads"),
        ),
        IoOp::SyncRead(false) => (
            "read amount is not handled",
            Some("use `Read::read_exact` instead, or handle partial reads"),
        ),
        IoOp::SyncWrite(false) => (
            "written amount is not handled",
            Some("use `Write::write_all` instead, or handle partial writes"),
        ),
        IoOp::AsyncWrite(false) => (
            "written amount is not handled",
            Some("use `AsyncWriteExt::write_all` instead, or handle partial writes"),
        ),
        IoOp::SyncRead(true) | IoOp::AsyncRead(true) => ("read amount is not handled", None),
        IoOp::SyncWrite(true) | IoOp::AsyncWrite(true) => ("written amount is not handled", None),
    };

    span_lint_hir_and_then(cx, UNUSED_IO_AMOUNT, at, span, msg, |diag| {
        if let Some(help_str) = help {
            diag.help(help_str);
        }
        for span in wild_cards {
            diag.span_note(
                *span,
                "the result is consumed here, but the amount of I/O bytes remains unhandled",
            );
        }
    });
}
