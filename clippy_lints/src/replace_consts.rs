use crate::utils::{match_def_path, span_lint_and_sugg};
use if_chain::if_chain;
use rustc::hir;
use rustc::hir::def::{DefKind, Res};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `ATOMIC_X_INIT`, `ONCE_INIT`, and
    /// `uX/iX::MIN/MAX`.
    ///
    /// **Why is this bad?** `const fn`s exist
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// static FOO: AtomicIsize = ATOMIC_ISIZE_INIT;
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust
    /// static FOO: AtomicIsize = AtomicIsize::new(0);
    /// ```
    pub REPLACE_CONSTS,
    pedantic,
    "Lint usages of standard library `const`s that could be replaced by `const fn`s"
}

declare_lint_pass!(ReplaceConsts => [REPLACE_CONSTS]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ReplaceConsts {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        if_chain! {
            if let hir::ExprKind::Path(ref qp) = expr.node;
            if let Res::Def(DefKind::Const, def_id) = cx.tables.qpath_res(qp, expr.hir_id);
            then {
                for (const_path, repl_snip) in &REPLACEMENTS {
                    if match_def_path(cx, def_id, const_path) {
                        span_lint_and_sugg(
                            cx,
                            REPLACE_CONSTS,
                            expr.span,
                            &format!("using `{}`", const_path.last().expect("empty path")),
                            "try this",
                            repl_snip.to_string(),
                            Applicability::MachineApplicable,
                        );
                        return;
                    }
                }
            }
        }
    }
}

const REPLACEMENTS: [([&str; 3], &str); 24] = [
    // Min
    (["core", "isize", "MIN"], "isize::min_value()"),
    (["core", "i8", "MIN"], "i8::min_value()"),
    (["core", "i16", "MIN"], "i16::min_value()"),
    (["core", "i32", "MIN"], "i32::min_value()"),
    (["core", "i64", "MIN"], "i64::min_value()"),
    (["core", "i128", "MIN"], "i128::min_value()"),
    (["core", "usize", "MIN"], "usize::min_value()"),
    (["core", "u8", "MIN"], "u8::min_value()"),
    (["core", "u16", "MIN"], "u16::min_value()"),
    (["core", "u32", "MIN"], "u32::min_value()"),
    (["core", "u64", "MIN"], "u64::min_value()"),
    (["core", "u128", "MIN"], "u128::min_value()"),
    // Max
    (["core", "isize", "MAX"], "isize::max_value()"),
    (["core", "i8", "MAX"], "i8::max_value()"),
    (["core", "i16", "MAX"], "i16::max_value()"),
    (["core", "i32", "MAX"], "i32::max_value()"),
    (["core", "i64", "MAX"], "i64::max_value()"),
    (["core", "i128", "MAX"], "i128::max_value()"),
    (["core", "usize", "MAX"], "usize::max_value()"),
    (["core", "u8", "MAX"], "u8::max_value()"),
    (["core", "u16", "MAX"], "u16::max_value()"),
    (["core", "u32", "MAX"], "u32::max_value()"),
    (["core", "u64", "MAX"], "u64::max_value()"),
    (["core", "u128", "MAX"], "u128::max_value()"),
];
