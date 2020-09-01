use crate::utils::span_lint_and_sugg;
use rustc_errors::Applicability;
use rustc_hir::{CaptureBy, Expr, ExprKind, PatKind, QPath};
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
    ///
    /// ```rust
    /// enum Errors {
    ///    Ignore
    ///}
    ///fn main() -> Result<(), Errors> {
    ///
    ///    let x = u32::try_from(-123_i32);
    ///
    ///    println!("{:?}", x.map_err(|_| Errors::Ignore));
    ///
    ///    Ok(())
    ///}
    /// ```
    /// Use instead:
    /// ```rust
    /// enum Errors {
    ///    WithContext(TryFromIntError)
    ///}
    ///fn main() -> Result<(), Errors> {
    ///
    ///    let x = u32::try_from(-123_i32);
    ///
    ///    println!("{:?}", x.map_err(|e| Errors::WithContext(e)));
    ///
    ///    Ok(())
    ///}
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
                                // Check the value of the closure to see if we can build the enum we are throwing away
                                // the error for make sure this is a Path
                                if let ExprKind::Path(q_path) = &closure_body.value.kind {
                                    // this should be a resolved path, only keep the path field
                                    if let QPath::Resolved(_, path) = q_path {
                                        // finally get the idents for each path segment collect them as a string and
                                        // join them with the path separator ("::"")
                                        let closure_fold: String = path
                                            .segments
                                            .iter()
                                            .map(|x| x.ident.as_str().to_string())
                                            .collect::<Vec<String>>()
                                            .join("::");
                                        //Span the body of the closure (the |...| bit) and suggest the fix by taking
                                        // the error and encapsulating it in the enum
                                        span_lint_and_sugg(
                                            cx,
                                            MAP_ERR_IGNORE,
                                            body_span,
                                            "`map_err` has thrown away the original error",
                                            "Allow the error enum to encapsulate the original error",
                                            format!("|e| {}(e)", closure_fold),
                                            Applicability::HasPlaceholders,
                                        );
                                    }
                                } else {
                                    //If we cannot build the enum in a human readable way just suggest not throwing way
                                    // the error
                                    span_lint_and_sugg(
                                        cx,
                                        MAP_ERR_IGNORE,
                                        body_span,
                                        "`map_err` has thrown away the original error",
                                        "Allow the error enum to encapsulate the original error",
                                        "|e|".to_string(),
                                        Applicability::HasPlaceholders,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
