use crate::utils::span_lint;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Symbol;

declare_clippy_lint! {
    /// **What it does:** Lints for specific trait methods defined in clippy.toml
    ///
    /// **Why is this bad?** Some methods are undesirable in certain contexts,
    /// and it would be beneficial to lint for them as needed.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust,ignore
    /// // example code where clippy issues a warning
    /// foo.bad_method(); // Foo::bad_method is disallowed in the configuration
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// // example code which does not raise clippy warning
    /// goodStruct.bad_method(); // GoodStruct::bad_method is not disallowed
    /// ```
    pub DISALLOWED_METHOD,
    nursery,
    "use of a disallowed method call"
}

#[derive(Clone, Debug)]
pub struct DisallowedMethod {
    disallowed: FxHashSet<Vec<Symbol>>,
}

impl DisallowedMethod {
    pub fn new(disallowed: &FxHashSet<String>) -> Self {
        Self {
            disallowed: disallowed
                .iter()
                .map(|s| s.split("::").map(|seg| Symbol::intern(seg)).collect::<Vec<_>>())
                .collect(),
        }
    }
}

impl_lint_pass!(DisallowedMethod => [DISALLOWED_METHOD]);

impl<'tcx> LateLintPass<'tcx> for DisallowedMethod {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::MethodCall(_path, _, _args, _) = &expr.kind {
            let def_id = cx.typeck_results().type_dependent_def_id(expr.hir_id).unwrap();

            let method_call = cx.get_def_path(def_id);
            if self.disallowed.contains(&method_call) {
                let method = method_call
                    .iter()
                    .map(|s| s.to_ident_string())
                    .collect::<Vec<_>>()
                    .join("::");

                span_lint(
                    cx,
                    DISALLOWED_METHOD,
                    expr.span,
                    &format!("use of a disallowed method `{}`", method),
                );
            }
        }
    }
}
