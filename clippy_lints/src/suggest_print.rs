use rustc::hir::*;
use rustc::lint::*;
use utils::{is_expn_of, match_def_path, resolve_node, span_lint};
use utils::opt_def_id;

/// **What it does:** Checks for usage of `write!()` / `writeln()!` which can be
/// replaced with `(e)print!()` / `(e)println!()`
///
/// **Why is this bad?** Using `(e)println! is clearer and more concise
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// // this would be clearer as `eprintln!("foo: {:?}", bar);`
/// writeln!(&mut io::stderr(), "foo: {:?}", bar).unwrap();
/// ```
declare_lint! {
    pub SUGGEST_PRINT,
    Warn,
    "using `write!()` family of functions instead of `print!()` family of \
     functions"
}

#[derive(Copy, Clone, Debug)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(SUGGEST_PRINT)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_let_chain! {[
            // match call to unwrap
            let ExprMethodCall(ref unwrap_fun, _, ref unwrap_args) = expr.node,
            unwrap_fun.name == "unwrap",
            // match call to write_fmt
            unwrap_args.len() > 0,
            let ExprMethodCall(ref write_fun, _, ref write_args) =
                unwrap_args[0].node,
            write_fun.name == "write_fmt",
            // match calls to std::io::stdout() / std::io::stderr ()
            write_args.len() > 0,
            let ExprCall(ref dest_fun, _) = write_args[0].node,
            let ExprPath(ref qpath) = dest_fun.node,
            let Some(dest_fun_id) =
                opt_def_id(resolve_node(cx, qpath, dest_fun.hir_id)),
            let Some(dest_name) = if match_def_path(cx.tcx, dest_fun_id, &["std", "io", "stdio", "stdout"]) {
                Some("stdout")
            } else if match_def_path(cx.tcx, dest_fun_id, &["std", "io", "stdio", "stderr"]) {
                Some("stderr")
            } else {
                None
            },
        ], {
            let dest_expr = &write_args[0];
            let (span, calling_macro) =
                if let Some(span) = is_expn_of(dest_expr.span, "write") {
                    (span, Some("write"))
                } else if let Some(span) = is_expn_of(dest_expr.span, "writeln") {
                    (span, Some("writeln"))
                } else {
                    (dest_expr.span, None)
                };
            let prefix = if dest_name == "stderr" {
                "e"
            } else {
                ""
            };
            if let Some(macro_name) = calling_macro {
                span_lint(
                    cx,
                    SUGGEST_PRINT,
                    span,
                    &format!(
                        "use of `{}!({}, ...).unwrap()`. Consider using `{}{}!` instead",
                        macro_name,
                        dest_name,
                        prefix,
                        macro_name.replace("write", "print")
                    )
                );
            } else {
                span_lint(
                    cx,
                    SUGGEST_PRINT,
                    span,
                    &format!(
                        "use of `{}.write_fmt(...).unwrap()`. Consider using `{}print!` instead",
                        dest_name,
                        prefix,
                    )
                );
            }
        }}
    }
}
