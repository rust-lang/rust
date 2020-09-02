use crate::utils::span_lint_and_help;

use rustc_hir::{CaptureBy, Expr, ExprKind, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for instances of `map_err(|_| Some::Enum)`
    ///
    /// **Why is this bad?** This map_err throws away the original error rather than allowing the enum to bubble the original error
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// Before:
    /// ```rust
    /// use std::convert::TryFrom;
    ///
    /// #[derive(Debug)]
    /// enum Errors {
    ///     Ignored
    /// }
    ///
    /// fn divisible_by_3(inp: i32) -> Result<u32, Errors> {
    ///     let i = u32::try_from(inp).map_err(|_| Errors::Ignored)?;
    ///
    ///     Ok(i)
    /// }
    ///  ```
    ///
    ///  After:
    ///  ```rust
    /// use std::convert::TryFrom;
    /// use std::num::TryFromIntError;
    /// use std::fmt;
    /// use std::error::Error;
    ///
    /// #[derive(Debug)]
    /// enum ParseError {
    ///     Indivisible {
    ///         source: TryFromIntError,
    ///         input: String,
    ///     }
    /// }
    ///
    /// impl fmt::Display for ParseError {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         match &self {
    ///             ParseError::Indivisible{source: _, input} => write!(f, "Error: {}", input)
    ///         }
    ///     }
    /// }
    ///
    /// impl Error for ParseError {}
    ///
    /// impl ParseError {
    ///     fn new(source: TryFromIntError, input: String) -> ParseError {
    ///         ParseError::Indivisible{source, input}
    ///     }
    /// }
    ///
    /// fn divisible_by_3(inp: i32) -> Result<u32, ParseError> {
    ///     let i = u32::try_from(inp).map_err(|e| ParseError::new(e, e.to_string()))?;
    ///
    ///     Ok(i)
    /// }
    /// ```
    pub MAP_ERR_IGNORE,
    style,
    "`map_err` should not ignore the original error"
}

declare_lint_pass!(MapErrIgnore => [MAP_ERR_IGNORE]);

impl<'tcx> LateLintPass<'tcx> for MapErrIgnore {
    // do not try to lint if this is from a macro or desugaring
    fn check_expr(&mut self, cx: &LateContext<'_>, e: &Expr<'_>) {
        if e.span.from_expansion() {
            return;
        }

        // check if this is a method call (e.g. x.foo())
        if let ExprKind::MethodCall(ref method, _t_span, ref args, _) = e.kind {
            // only work if the method name is `map_err` and there are only 2 arguments (e.g. x.map_err(|_|[1]
            // Enum::Variant[2]))
            if method.ident.as_str() == "map_err" && args.len() == 2 {
                // make sure the first argument is a closure, and grab the CaptureRef, body_id, and body_span fields
                if let ExprKind::Closure(capture, _, body_id, body_span, _) = args[1].kind {
                    // check if this is by Reference (meaning there's no move statement)
                    if capture == CaptureBy::Ref {
                        // Get the closure body to check the parameters and values
                        let closure_body = cx.tcx.hir().body(body_id);
                        // make sure there's only one parameter (`|_|`)
                        if closure_body.params.len() == 1 {
                            // make sure that parameter is the wild token (`_`)
                            if let PatKind::Wild = closure_body.params[0].pat.kind {
                                // span the area of the closure capture and warn that the
                                // original error will be thrown away
                                span_lint_and_help(
                                    cx,
                                    MAP_ERR_IGNORE,
                                    body_span,
                                    "`map_else(|_|...` ignores the original error",
                                    None,
                                    "Consider wrapping the error in an enum variant",
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
