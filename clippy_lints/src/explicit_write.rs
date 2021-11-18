use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg};
use clippy_utils::higher::FormatArgsExpn;
use clippy_utils::{is_expn_of, match_function_call, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `write!()` / `writeln()!` which can be
    /// replaced with `(e)print!()` / `(e)println!()`
    ///
    /// ### Why is this bad?
    /// Using `(e)println! is clearer and more concise
    ///
    /// ### Example
    /// ```rust
    /// # use std::io::Write;
    /// # let bar = "furchtbar";
    /// // this would be clearer as `eprintln!("foo: {:?}", bar);`
    /// writeln!(&mut std::io::stderr(), "foo: {:?}", bar).unwrap();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub EXPLICIT_WRITE,
    complexity,
    "using the `write!()` family of functions instead of the `print!()` family of functions, when using the latter would work"
}

declare_lint_pass!(ExplicitWrite => [EXPLICIT_WRITE]);

impl<'tcx> LateLintPass<'tcx> for ExplicitWrite {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            // match call to unwrap
            if let ExprKind::MethodCall(unwrap_fun, _, [write_call], _) = expr.kind;
            if unwrap_fun.ident.name == sym::unwrap;
            // match call to write_fmt
            if let ExprKind::MethodCall(write_fun, _, [write_recv, write_arg], _) = write_call.kind;
            if write_fun.ident.name == sym!(write_fmt);
            // match calls to std::io::stdout() / std::io::stderr ()
            if let Some(dest_name) = if match_function_call(cx, write_recv, &paths::STDOUT).is_some() {
                Some("stdout")
            } else if match_function_call(cx, write_recv, &paths::STDERR).is_some() {
                Some("stderr")
            } else {
                None
            };
            if let Some(format_args) = FormatArgsExpn::parse(write_arg);
            then {
                let calling_macro =
                    // ordering is important here, since `writeln!` uses `write!` internally
                    if is_expn_of(write_call.span, "writeln").is_some() {
                        Some("writeln")
                    } else if is_expn_of(write_call.span, "write").is_some() {
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
                let (used, sugg_mac) = if let Some(macro_name) = calling_macro {
                    (
                        format!("{}!({}(), ...)", macro_name, dest_name),
                        macro_name.replace("write", "print"),
                    )
                } else {
                    (
                        format!("{}().write_fmt(...)", dest_name),
                        "print".into(),
                    )
                };
                let msg = format!("use of `{}.unwrap()`", used);
                if let [write_output] = *format_args.format_string_symbols {
                    let mut write_output = write_output.to_string();
                    if write_output.ends_with('\n') {
                        write_output.pop();
                    }

                    let sugg = format!("{}{}!(\"{}\")", prefix, sugg_mac, write_output.escape_default());
                    span_lint_and_sugg(
                        cx,
                        EXPLICIT_WRITE,
                        expr.span,
                        &msg,
                        "try this",
                        sugg,
                        Applicability::MachineApplicable
                    );
                } else {
                    // We don't have a proper suggestion
                    let help = format!("consider using `{}{}!` instead", prefix, sugg_mac);
                    span_lint_and_help(cx, EXPLICIT_WRITE, expr.span, &msg, None, &help);
                }
            }
        }
    }
}
