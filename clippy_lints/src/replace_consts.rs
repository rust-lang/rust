use crate::utils::sym;
use crate::utils::{match_def_path, span_lint_and_sugg};
use if_chain::if_chain;
use lazy_static::lazy_static;
use rustc::hir;
use rustc::hir::def::{DefKind, Res};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::symbol::Symbol;

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
                for (const_path, repl_snip) in REPLACEMENTS.iter() {
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

lazy_static! {
static ref REPLACEMENTS: [([Symbol; 3], &'static str); 25] = [
    // Once
    ([*sym::core, *sym::sync, *sym::ONCE_INIT], "Once::new()"),
    // Min
    ([*sym::core, *sym::isize, *sym::MIN], "isize::min_value()"),
    ([*sym::core, *sym::i8, *sym::MIN], "i8::min_value()"),
    ([*sym::core, *sym::i16, *sym::MIN], "i16::min_value()"),
    ([*sym::core, *sym::i32, *sym::MIN], "i32::min_value()"),
    ([*sym::core, *sym::i64, *sym::MIN], "i64::min_value()"),
    ([*sym::core, *sym::i128, *sym::MIN], "i128::min_value()"),
    ([*sym::core, *sym::usize, *sym::MIN], "usize::min_value()"),
    ([*sym::core, *sym::u8, *sym::MIN], "u8::min_value()"),
    ([*sym::core, *sym::u16, *sym::MIN], "u16::min_value()"),
    ([*sym::core, *sym::u32, *sym::MIN], "u32::min_value()"),
    ([*sym::core, *sym::u64, *sym::MIN], "u64::min_value()"),
    ([*sym::core, *sym::u128, *sym::MIN], "u128::min_value()"),
    // Max
    ([*sym::core, *sym::isize, *sym::MAX], "isize::max_value()"),
    ([*sym::core, *sym::i8, *sym::MAX], "i8::max_value()"),
    ([*sym::core, *sym::i16, *sym::MAX], "i16::max_value()"),
    ([*sym::core, *sym::i32, *sym::MAX], "i32::max_value()"),
    ([*sym::core, *sym::i64, *sym::MAX], "i64::max_value()"),
    ([*sym::core, *sym::i128, *sym::MAX], "i128::max_value()"),
    ([*sym::core, *sym::usize, *sym::MAX], "usize::max_value()"),
    ([*sym::core, *sym::u8, *sym::MAX], "u8::max_value()"),
    ([*sym::core, *sym::u16, *sym::MAX], "u16::max_value()"),
    ([*sym::core, *sym::u32, *sym::MAX], "u32::max_value()"),
    ([*sym::core, *sym::u64, *sym::MAX], "u64::max_value()"),
    ([*sym::core, *sym::u128, *sym::MAX], "u128::max_value()"),
];
}
