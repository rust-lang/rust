use crate::utils::{match_type, span_lint_and_sugg};
use crate::utils::paths;
use crate::utils::sugg::Sugg;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_lint::{LateLintPass, LateContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_hir::*;

declare_clippy_lint! {
    /// **What it does:**
    /// Detects when people use `Vec::sort_by` and pass in a function
    /// which compares the second argument to the first.
    ///
    /// **Why is this bad?**
    /// It is more clear to use `Vec::sort_by_key` and `std::cmp::Reverse`
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// vec.sort_by(|a, b| b.foo().cmp(&a.foo()));
    /// ```
    /// Use instead:
    /// ```rust
    /// vec.sort_by_key(|e| Reverse(e.foo()));
    /// ```
    pub SORT_BY_KEY_REVERSE,
    complexity,
    "Use of `Vec::sort_by` when `Vec::sort_by_key` would be clearer"
}

declare_lint_pass!(SortByKeyReverse => [SORT_BY_KEY_REVERSE]);

struct LintTrigger {
    vec_name: String,
    closure_arg: String,
    closure_reverse_body: String,
    unstable: bool,
}

fn detect_lint(cx: &LateContext<'_, '_>, expr: &Expr<'_>) -> Option<LintTrigger> {
    if_chain! {
        if let ExprKind::MethodCall(name_ident, _, args) = &expr.kind;
        if let name = name_ident.ident.name.to_ident_string();
        if name == "sort_by" || name == "sort_unstable_by";
        if let [vec, Expr { kind: ExprKind::Closure(_, closure_decl, closure_body_id, _, _), .. }] = args;
        if closure_decl.inputs.len() == 2;
        if match_type(cx, &cx.tables.expr_ty(vec), &paths::VEC);
        then {
            let vec_name = Sugg::hir(cx, &args[0], "..").to_string();
            let unstable = name == "sort_unstable_by";
            Some(LintTrigger { vec_name, unstable, closure_arg: "e".to_string(), closure_reverse_body: "e".to_string() })
        } else {
            None
        }
    }
}

impl LateLintPass<'_, '_> for SortByKeyReverse {
    fn check_expr(&mut self, cx: &LateContext<'_, '_>, expr: &Expr<'_>) {
        println!("{:?}", expr);
        span_lint_and_sugg(
            cx,
            SORT_BY_KEY_REVERSE,
            expr.span,
            "use Vec::sort_by_key here instead",
            "try",
            String::from("being a better person"),
            Applicability::MachineApplicable,
        );
        if let Some(trigger) = detect_lint(cx, expr) {
            span_lint_and_sugg(
                cx,
                SORT_BY_KEY_REVERSE,
                expr.span,
                "use Vec::sort_by_key here instead",
                "try",
                format!(
                    "{}.sort{}_by_key(|{}| Reverse({}))",
                    trigger.vec_name,
                    if trigger.unstable { "_unstable" } else { "" },
                    trigger.closure_arg,
                    trigger.closure_reverse_body,
                ),
                Applicability::MachineApplicable,
            );
        }
    }
}
