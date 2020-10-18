use crate::{LateContext, LateLintPass, LintContext};
use rustc_ast as ast;
use rustc_hir as hir;
use rustc_middle::ty;

declare_lint! {
    /// The `panic_fmt` lint detects `panic!("..")` with `{` or `}` in the string literal.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// panic!("{}");
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `panic!("{}")` panics with the message `"{}"`, as a `panic!()` invocation
    /// with a single argument does not use `format_args!()`.
    /// A future version of Rust will interpret this string as format string,
    /// which would break this.
    PANIC_FMT,
    Warn,
    "detect braces in single-argument panic!() invocations",
    report_in_external_macro
}

declare_lint_pass!(PanicFmt => [PANIC_FMT]);

impl<'tcx> LateLintPass<'tcx> for PanicFmt {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Call(f, [arg]) = &expr.kind {
            if let &ty::FnDef(def_id, _) = cx.typeck_results().expr_ty(f).kind() {
                if Some(def_id) == cx.tcx.lang_items().begin_panic_fn()
                    || Some(def_id) == cx.tcx.lang_items().panic_fn()
                {
                    check_panic(cx, f, arg);
                }
            }
        }
    }
}

fn check_panic<'tcx>(cx: &LateContext<'tcx>, f: &'tcx hir::Expr<'tcx>, arg: &'tcx hir::Expr<'tcx>) {
    if let hir::ExprKind::Lit(lit) = &arg.kind {
        if let ast::LitKind::Str(sym, _) = lit.node {
            if sym.as_str().contains(&['{', '}'][..]) {
                cx.struct_span_lint(PANIC_FMT, f.span, |lint| {
                    lint.build("Panic message contains a brace")
                    .note("This message is not used as a format string, but will be in a future Rust version")
                    .emit();
                });
            }
        }
    }
}
