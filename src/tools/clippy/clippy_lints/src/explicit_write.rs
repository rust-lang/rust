use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::{FormatArgsStorage, format_args_inputs_span};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_expn_of, path_def_id, sym};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{BindingMode, Block, BlockCheckMode, Expr, ExprKind, Node, PatKind, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::ExpnId;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `write!()` / `writeln()!` which can be
    /// replaced with `(e)print!()` / `(e)println!()`
    ///
    /// ### Why is this bad?
    /// Using `(e)println!` is clearer and more concise
    ///
    /// ### Example
    /// ```no_run
    /// # use std::io::Write;
    /// # let bar = "furchtbar";
    /// writeln!(&mut std::io::stderr(), "foo: {:?}", bar).unwrap();
    /// writeln!(&mut std::io::stdout(), "foo: {:?}", bar).unwrap();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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

pub struct ExplicitWrite {
    format_args: FormatArgsStorage,
}

impl ExplicitWrite {
    pub fn new(format_args: FormatArgsStorage) -> Self {
        Self { format_args }
    }
}

impl_lint_pass!(ExplicitWrite => [EXPLICIT_WRITE]);

impl<'tcx> LateLintPass<'tcx> for ExplicitWrite {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // match call to unwrap
        if let ExprKind::MethodCall(unwrap_fun, write_call, [], _) = expr.kind
            && unwrap_fun.ident.name == sym::unwrap
            // match call to write_fmt
            && let ExprKind::MethodCall(write_fun, write_recv, [write_arg], _) = *look_in_block(cx, &write_call.kind)
            && let ExprKind::Call(write_recv_path, []) = write_recv.kind
            && write_fun.ident.name == sym::write_fmt
            && let Some(def_id) = path_def_id(cx, write_recv_path)
        {
            // match calls to std::io::stdout() / std::io::stderr ()
            let (dest_name, prefix) = match cx.tcx.get_diagnostic_name(def_id) {
                Some(sym::io_stdout) => ("stdout", ""),
                Some(sym::io_stderr) => ("stderr", "e"),
                _ => return,
            };
            let Some(format_args) = self.format_args.get(cx, write_arg, ExpnId::root()) else {
                return;
            };

            // ordering is important here, since `writeln!` uses `write!` internally
            let calling_macro = if is_expn_of(write_call.span, sym::writeln).is_some() {
                Some("writeln")
            } else if is_expn_of(write_call.span, sym::write).is_some() {
                Some("write")
            } else {
                None
            };

            // We need to remove the last trailing newline from the string because the
            // underlying `fmt::write` function doesn't know whether `println!` or `print!` was
            // used.
            let (used, sugg_mac) = if let Some(macro_name) = calling_macro {
                (
                    format!("{macro_name}!({dest_name}(), ...)"),
                    macro_name.replace("write", "print"),
                )
            } else {
                (format!("{dest_name}().write_fmt(...)"), "print".into())
            };
            let mut applicability = Applicability::MachineApplicable;
            let inputs_snippet =
                snippet_with_applicability(cx, format_args_inputs_span(format_args), "..", &mut applicability);
            span_lint_and_sugg(
                cx,
                EXPLICIT_WRITE,
                expr.span,
                format!("use of `{used}.unwrap()`"),
                "try",
                format!("{prefix}{sugg_mac}!({inputs_snippet})"),
                applicability,
            );
        }
    }
}

/// If `kind` is a block that looks like `{ let result = $expr; result }` then
/// returns $expr. Otherwise returns `kind`.
fn look_in_block<'tcx, 'hir>(cx: &LateContext<'tcx>, kind: &'tcx ExprKind<'hir>) -> &'tcx ExprKind<'hir> {
    if let ExprKind::Block(block, _label @ None) = kind
        && let Block {
            stmts: [Stmt { kind: StmtKind::Let(local), .. }],
            expr: Some(expr_end_of_block),
            rules: BlockCheckMode::DefaultBlock,
            ..
        } = block

        // Find id of the local that expr_end_of_block resolves to
        && let ExprKind::Path(QPath::Resolved(None, expr_path)) = expr_end_of_block.kind
        && let Res::Local(expr_res) = expr_path.res
        && let Node::Pat(res_pat) = cx.tcx.hir_node(expr_res)

        // Find id of the local we found in the block
        && let PatKind::Binding(BindingMode::NONE, local_hir_id, _ident, None) = local.pat.kind

        // If those two are the same hir id
        && res_pat.hir_id == local_hir_id

        && let Some(init) = local.init
    {
        return &init.kind;
    }
    kind
}
