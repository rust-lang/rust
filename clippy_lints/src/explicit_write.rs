use crate::utils::{is_expn_of, match_def_path, paths, resolve_node, span_lint, span_lint_and_sugg};
use if_chain::if_chain;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::ast::LitKind;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `write!()` / `writeln()!` which can be
    /// replaced with `(e)print!()` / `(e)println!()`
    ///
    /// **Why is this bad?** Using `(e)println! is clearer and more concise
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # use std::io::Write;
    /// # let bar = "furchtbar";
    /// // this would be clearer as `eprintln!("foo: {:?}", bar);`
    /// writeln!(&mut std::io::stderr(), "foo: {:?}", bar).unwrap();
    /// ```
    pub EXPLICIT_WRITE,
    complexity,
    "using the `write!()` family of functions instead of the `print!()` family of functions, when using the latter would work"
}

declare_lint_pass!(ExplicitWrite => [EXPLICIT_WRITE]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ExplicitWrite {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            // match call to unwrap
            if let ExprKind::MethodCall(ref unwrap_fun, _, ref unwrap_args) = expr.node;
            if unwrap_fun.ident.name == sym!(unwrap);
            // match call to write_fmt
            if unwrap_args.len() > 0;
            if let ExprKind::MethodCall(ref write_fun, _, ref write_args) =
                unwrap_args[0].node;
            if write_fun.ident.name == sym!(write_fmt);
            // match calls to std::io::stdout() / std::io::stderr ()
            if write_args.len() > 0;
            if let ExprKind::Call(ref dest_fun, _) = write_args[0].node;
            if let ExprKind::Path(ref qpath) = dest_fun.node;
            if let Some(dest_fun_id) = resolve_node(cx, qpath, dest_fun.hir_id).opt_def_id();
            if let Some(dest_name) = if match_def_path(cx, dest_fun_id, &paths::STDOUT) {
                Some("stdout")
            } else if match_def_path(cx, dest_fun_id, &paths::STDERR) {
                Some("stderr")
            } else {
                None
            };
            then {
                let write_span = unwrap_args[0].span;
                let calling_macro =
                    // ordering is important here, since `writeln!` uses `write!` internally
                    if is_expn_of(write_span, "writeln").is_some() {
                        Some("writeln")
                    } else if is_expn_of(write_span, "write").is_some() {
                        Some("write")
                    } else {
                        None
                    };
                let prefix = if dest_name == "stderr" {
                    "e"
                } else {
                    ""
                };

                // We need to remove the last trailing newline from the string because the
                // underlying `fmt::write` function doesn't know whether `println!` or `print!` was
                // used.
                if let Some(mut write_output) = write_output_string(write_args) {
                    if write_output.ends_with('\n') {
                        write_output.pop();
                    }

                    if let Some(macro_name) = calling_macro {
                        span_lint_and_sugg(
                            cx,
                            EXPLICIT_WRITE,
                            expr.span,
                            &format!(
                                "use of `{}!({}(), ...).unwrap()`",
                                macro_name,
                                dest_name
                            ),
                            "try this",
                            format!("{}{}!(\"{}\")", prefix, macro_name.replace("write", "print"), write_output.escape_default()),
                            Applicability::MachineApplicable
                        );
                    } else {
                        span_lint_and_sugg(
                            cx,
                            EXPLICIT_WRITE,
                            expr.span,
                            &format!("use of `{}().write_fmt(...).unwrap()`", dest_name),
                            "try this",
                            format!("{}print!(\"{}\")", prefix, write_output.escape_default()),
                            Applicability::MachineApplicable
                        );
                    }
                } else {
                    // We don't have a proper suggestion
                    if let Some(macro_name) = calling_macro {
                        span_lint(
                            cx,
                            EXPLICIT_WRITE,
                            expr.span,
                            &format!(
                                "use of `{}!({}(), ...).unwrap()`. Consider using `{}{}!` instead",
                                macro_name,
                                dest_name,
                                prefix,
                                macro_name.replace("write", "print")
                            )
                        );
                    } else {
                        span_lint(
                            cx,
                            EXPLICIT_WRITE,
                            expr.span,
                            &format!("use of `{}().write_fmt(...).unwrap()`. Consider using `{}print!` instead", dest_name, prefix),
                        );
                    }
                }

            }
        }
    }
}

// Extract the output string from the given `write_args`.
fn write_output_string(write_args: &HirVec<Expr>) -> Option<String> {
    if_chain! {
        // Obtain the string that should be printed
        if write_args.len() > 1;
        if let ExprKind::Call(_, ref output_args) = write_args[1].node;
        if output_args.len() > 0;
        if let ExprKind::AddrOf(_, ref output_string_expr) = output_args[0].node;
        if let ExprKind::Array(ref string_exprs) = output_string_expr.node;
        if string_exprs.len() > 0;
        if let ExprKind::Lit(ref lit) = string_exprs[0].node;
        if let LitKind::Str(ref write_output, _) = lit.node;
        then {
            return Some(write_output.to_string())
        }
    }
    None
}
