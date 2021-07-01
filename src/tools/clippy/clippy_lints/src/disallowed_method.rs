use clippy_utils::diagnostics::span_lint;
use clippy_utils::fn_def_id;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::{def::Res, def_id::DefId, Crate, Expr};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Symbol;

declare_clippy_lint! {
    /// **What it does:** Denies the configured methods and functions in clippy.toml
    ///
    /// **Why is this bad?** Some methods are undesirable in certain contexts,
    /// and it's beneficial to lint for them as needed.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-methods = ["std::vec::Vec::leak", "std::time::Instant::now"]
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
    def_ids: FxHashSet<(DefId, Vec<Symbol>)>,
}

impl DisallowedMethod {
    pub fn new(disallowed: &FxHashSet<String>) -> Self {
        Self {
            disallowed: disallowed
                .iter()
                .map(|s| s.split("::").map(|seg| Symbol::intern(seg)).collect::<Vec<_>>())
                .collect(),
            def_ids: FxHashSet::default(),
        }
    }
}

impl_lint_pass!(DisallowedMethod => [DISALLOWED_METHOD]);

impl<'tcx> LateLintPass<'tcx> for DisallowedMethod {
    fn check_crate(&mut self, cx: &LateContext<'_>, _: &Crate<'_>) {
        for path in &self.disallowed {
            let segs = path.iter().map(ToString::to_string).collect::<Vec<_>>();
            if let Res::Def(_, id) = clippy_utils::path_to_res(cx, &segs.iter().map(String::as_str).collect::<Vec<_>>())
            {
                self.def_ids.insert((id, path.clone()));
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let Some(def_id) = fn_def_id(cx, expr) {
            if self.def_ids.iter().any(|(id, _)| def_id == *id) {
                let func_path = cx.get_def_path(def_id);
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
