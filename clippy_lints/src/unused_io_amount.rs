use rustc::lint::*;
use rustc::hir;
use utils::{span_lint, match_path, match_trait_method, paths};

/// **What it does:** Checks for unused written/read amount.
///
/// **Why is this bad?** `io::Write::write` and `io::Read::read` are not guaranteed to
/// process the entire buffer. They return how many bytes were processed, which might be smaller
/// than a given buffer's length. If you don't need to deal with partial-write/read, use
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
            hir::StmtSemi(ref expr, _) |
            hir::StmtExpr(ref expr, _) => &**expr,
            _ => return,
        };

        if let hir::ExprRet(..) = expr.node {
            return;
        }

        match expr.node {
            hir::ExprMatch(ref expr, ref arms, _) if is_try(arms) => {
                if let hir::ExprCall(ref func, ref args) = expr.node {
                    if let hir::ExprPath(ref path) = func.node {
                        if match_path(path, &paths::CARRIER_TRANSLATE) && args.len() == 1 {
                            check_method_call(cx, &args[0], expr);
                        }
                    }
                } else {
                    check_method_call(cx, expr, expr);
                }
            },

            hir::ExprMethodCall(ref symbol, _, ref args) => {
                let symbol = &*symbol.node.as_str();
                match symbol {
                    "expect" | "unwrap" | "unwrap_or" | "unwrap_or_else" => {
                        check_method_call(cx, &args[0], expr);
                    },
                    _ => (),
                }
            },

            _ => (),
        }
    }
}

fn check_method_call(cx: &LateContext, call: &hir::Expr, expr: &hir::Expr) {
    if let hir::ExprMethodCall(ref symbol, _, _) = call.node {
        let symbol = &*symbol.node.as_str();
        if match_trait_method(cx, call, &paths::IO_READ) && symbol == "read" {
            span_lint(cx,
                      UNUSED_IO_AMOUNT,
                      expr.span,
                      "handle read amount returned or use `Read::read_exact` instead");
        } else if match_trait_method(cx, call, &paths::IO_WRITE) && symbol == "write" {
            span_lint(cx,
                      UNUSED_IO_AMOUNT,
                      expr.span,
                      "handle written amount returned or use `Write::write_all` instead");
        }
    }
}

fn is_try(arms: &[hir::Arm]) -> bool {
    // `Ok(x) => x` or `Ok(_) => ...`
    fn is_ok(arm: &hir::Arm) -> bool {
        if let hir::PatKind::TupleStruct(ref path, ref pat, ref dotdot) = arm.pats[0].node {
            // cut off `core`
            if match_path(path, &paths::RESULT_OK[1..]) {
                if *dotdot == Some(0) {
                    return true;
                }

                match pat[0].node {
                    hir::PatKind::Wild => {
                        return true;
                    },
                    hir::PatKind::Binding(_, defid, _, None) => {
                        if let hir::ExprPath(hir::QPath::Resolved(None, ref path)) = arm.body.node {
                            if path.def.def_id() == defid {
                                return true;
                            }
                        }
                    },
                    _ => (),
                }
            }
        }

        false
    }

    /// Detects `_ => ...` or `Err(x) => ...`
    fn is_err_or_wild(arm: &hir::Arm) -> bool {
        match arm.pats[0].node {
            hir::PatKind::Wild => true,
            hir::PatKind::TupleStruct(ref path, _, _) => match_path(path, &paths::RESULT_ERR[1..]),
            _ => false,
        }
    }

    if arms.len() == 2 && arms[0].pats.len() == 1 && arms[0].guard.is_none() && arms[1].pats.len() == 1 &&
       arms[1].guard.is_none() {
        (is_ok(&arms[0]) && is_err_or_wild(&arms[1])) || (is_ok(&arms[1]) && is_err_or_wild(&arms[0]))
    } else {
        false
    }
}
