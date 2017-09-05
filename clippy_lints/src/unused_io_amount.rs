use rustc::lint::*;
use rustc::hir;
use utils::{is_try, match_qpath, match_trait_method, paths, span_lint};

/// **What it does:** Checks for unused written/read amount.
///
/// **Why is this bad?** `io::Write::write` and `io::Read::read` are not
/// guaranteed to
/// process the entire buffer. They return how many bytes were processed, which
/// might be smaller
/// than a given buffer's length. If you don't need to deal with
/// partial-write/read, use
/// `write_all`/`read_exact` instead.
///
/// **Known problems:** Detects only common patterns.
///
/// **Example:**
/// ```rust,ignore
/// use std::io;
/// fn foo<W: io::Write>(w: &mut W) -> io::Result<()> {
///     // must be `w.write_all(b"foo")?;`
///     w.write(b"foo")?;
///     Ok(())
/// }
/// ```
declare_lint! {
    pub UNUSED_IO_AMOUNT,
    Deny,
    "unused written/read amount"
}

pub struct UnusedIoAmount;

impl LintPass for UnusedIoAmount {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_IO_AMOUNT)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnusedIoAmount {
    fn check_stmt(&mut self, cx: &LateContext, s: &hir::Stmt) {
        let expr = match s.node {
            hir::StmtSemi(ref expr, _) | hir::StmtExpr(ref expr, _) => &**expr,
            _ => return,
        };

        match expr.node {
            hir::ExprMatch(ref res, _, _) if is_try(expr).is_some() => {
                if let hir::ExprCall(ref func, ref args) = res.node {
                    if let hir::ExprPath(ref path) = func.node {
                        if match_qpath(path, &paths::TRY_INTO_RESULT) && args.len() == 1 {
                            check_method_call(cx, &args[0], expr);
                        }
                    }
                } else {
                    check_method_call(cx, res, expr);
                }
            },

            hir::ExprMethodCall(ref path, _, ref args) => match &*path.name.as_str() {
                "expect" | "unwrap" | "unwrap_or" | "unwrap_or_else" => {
                    check_method_call(cx, &args[0], expr);
                },
                _ => (),
            },

            _ => (),
        }
    }
}

fn check_method_call(cx: &LateContext, call: &hir::Expr, expr: &hir::Expr) {
    if let hir::ExprMethodCall(ref path, _, _) = call.node {
        let symbol = &*path.name.as_str();
        if match_trait_method(cx, call, &paths::IO_READ) && symbol == "read" {
            span_lint(
                cx,
                UNUSED_IO_AMOUNT,
                expr.span,
                "handle read amount returned or use `Read::read_exact` instead",
            );
        } else if match_trait_method(cx, call, &paths::IO_WRITE) && symbol == "write" {
            span_lint(
                cx,
                UNUSED_IO_AMOUNT,
                expr.span,
                "handle written amount returned or use `Write::write_all` instead",
            );
        }
    }
}
