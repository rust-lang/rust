use crate::utils::{match_def_path, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, ExprKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for usage of standard library
    /// `const`s that could be replaced by `const fn`s.
    ///
    /// **Why is this bad?** `const fn`s exist
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x = std::u32::MIN;
    /// let y = std::u32::MAX;
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust
    /// let x = u32::min_value();
    /// let y = u32::max_value();
    /// ```
    pub REPLACE_CONSTS,
    pedantic,
    "Lint usages of standard library `const`s that could be replaced by `const fn`s"
}

declare_lint_pass!(ReplaceConsts => [REPLACE_CONSTS]);

fn in_pattern(cx: &LateContext<'_, '_>, expr: &Expr<'_>) -> bool {
    let map = &cx.tcx.hir();
    let parent_id = map.get_parent_node(expr.hir_id);

    if let Some(node) = map.find(parent_id) {
        if let Node::Pat(_) = node {
            return true;
        }
    }

    false
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ReplaceConsts {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Path(ref qp) = expr.kind;
            if let Res::Def(DefKind::Const, def_id) = cx.tables.qpath_res(qp, expr.hir_id);
            // Do not lint within patterns as function calls are disallowed in them
            if !in_pattern(cx, expr);
            then {
                for &(ref const_path, repl_snip) in &REPLACEMENTS {
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
