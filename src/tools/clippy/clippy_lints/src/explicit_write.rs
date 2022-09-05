use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::FormatArgsExpn;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_expn_of, match_function_call, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{BindingAnnotation, Block, BlockCheckMode, Expr, ExprKind, Node, PatKind, QPath, Stmt, StmtKind};
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
    /// writeln!(&mut std::io::stderr(), "foo: {:?}", bar).unwrap();
    /// writeln!(&mut std::io::stdout(), "foo: {:?}", bar).unwrap();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # use std::io::Write;
    /// # let bar = "furchtbar";
    /// eprintln!("foo: {:?}", bar);
    /// println!("foo: {:?}", bar);
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
            if let ExprKind::MethodCall(unwrap_fun, write_call, [], _) = expr.kind;
            if unwrap_fun.ident.name == sym::unwrap;
            // match call to write_fmt
            if let ExprKind::MethodCall(write_fun, write_recv, [write_arg], _) = look_in_block(cx, &write_call.kind);
            if write_fun.ident.name == sym!(write_fmt);
            // match calls to std::io::stdout() / std::io::stderr ()
            if let Some(dest_name) = if match_function_call(cx, write_recv, &paths::STDOUT).is_some() {
                Some("stdout")
            } else if match_function_call(cx, write_recv, &paths::STDERR).is_some() {
                Some("stderr")
            } else {
                None
            };
            if let Some(format_args) = FormatArgsExpn::parse(cx, write_arg);
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
                let mut applicability = Applicability::MachineApplicable;
                let inputs_snippet = snippet_with_applicability(
                    cx,
                    format_args.inputs_span(),
                    "..",
                    &mut applicability,
                );
                span_lint_and_sugg(
                    cx,
                    EXPLICIT_WRITE,
                    expr.span,
                    &format!("use of `{}.unwrap()`", used),
                    "try this",
                    format!("{}{}!({})", prefix, sugg_mac, inputs_snippet),
                    applicability,
                )
            }
        }
    }
}

/// If `kind` is a block that looks like `{ let result = $expr; result }` then
/// returns $expr. Otherwise returns `kind`.
fn look_in_block<'tcx, 'hir>(cx: &LateContext<'tcx>, kind: &'tcx ExprKind<'hir>) -> &'tcx ExprKind<'hir> {
    if_chain! {
        if let ExprKind::Block(block, _label @ None) = kind;
        if let Block {
            stmts: [Stmt { kind: StmtKind::Local(local), .. }],
            expr: Some(expr_end_of_block),
            rules: BlockCheckMode::DefaultBlock,
            ..
        } = block;

        // Find id of the local that expr_end_of_block resolves to
        if let ExprKind::Path(QPath::Resolved(None, expr_path)) = expr_end_of_block.kind;
        if let Res::Local(expr_res) = expr_path.res;
        if let Some(Node::Pat(res_pat)) = cx.tcx.hir().find(expr_res);

        // Find id of the local we found in the block
        if let PatKind::Binding(BindingAnnotation::Unannotated, local_hir_id, _ident, None) = local.pat.kind;

        // If those two are the same hir id
        if res_pat.hir_id == local_hir_id;

        if let Some(init) = local.init;
        then {
            return &init.kind;
        }
    }
    kind
}
