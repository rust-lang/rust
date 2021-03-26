use clippy_utils::diagnostics::span_lint;
use clippy_utils::fn_def_id;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Symbol;

declare_clippy_lint! {
    /// **What it does:** Denies the configured methods and functions in clippy.toml
    ///
    /// **Why is this bad?** Some methods are undesirable in certain contexts,
    /// and it's beneficial to lint for them as needed.
    ///
    /// **Known problems:** Currently, you must write each function as a
    /// fully-qualified path. This lint doesn't support aliases or reexported
    /// names; be aware that many types in `std` are actually reexports.
    ///
    /// For example, if you want to disallow `Duration::as_secs`, your clippy.toml
    /// configuration would look like
    /// `disallowed-methods = ["core::time::Duration::as_secs"]` and not
    /// `disallowed-methods = ["std::time::Duration::as_secs"]` as you might expect.
    ///
    /// **Example:**
    ///
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-methods = ["alloc::vec::Vec::leak", "std::time::Instant::now"]
    /// ```
    ///
    /// ```rust,ignore
    /// // Example code where clippy issues a warning
    /// let xs = vec![1, 2, 3, 4];
    /// xs.leak(); // Vec::leak is disallowed in the config.
    ///
    /// let _now = Instant::now(); // Instant::now is disallowed in the config.
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// // Example code which does not raise clippy warning
    /// let mut xs = Vec::new(); // Vec::new is _not_ disallowed in the config.
    /// xs.push(123); // Vec::push is _not_ disallowed in the config.
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
        if let Some(def_id) = fn_def_id(cx, expr) {
            let func_path = cx.get_def_path(def_id);
            if self.disallowed.contains(&func_path) {
                let func_path_string = func_path
                    .into_iter()
                    .map(Symbol::to_ident_string)
                    .collect::<Vec<_>>()
                    .join("::");

                span_lint(
                    cx,
                    DISALLOWED_METHOD,
                    expr.span,
                    &format!("use of a disallowed method `{}`", func_path_string),
                );
            }
        }
    }
}
