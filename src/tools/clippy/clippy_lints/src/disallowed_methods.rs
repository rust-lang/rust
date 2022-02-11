use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::fn_def_id;

use rustc_hir::{def::Res, def_id::DefIdMap, Expr};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};

use crate::utils::conf;

declare_clippy_lint! {
    /// ### What it does
    /// Denies the configured methods and functions in clippy.toml
    ///
    /// Note: Even though this lint is warn-by-default, it will only trigger if
    /// methods are defined in the clippy.toml file.
    ///
    /// ### Why is this bad?
    /// Some methods are undesirable in certain contexts, and it's beneficial to
    /// lint for them as needed.
    ///
    /// ### Example
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-methods = [
    ///     # Can use a string as the path of the disallowed method.
    ///     "std::boxed::Box::new",
    ///     # Can also use an inline table with a `path` key.
    ///     { path = "std::time::Instant::now" },
    ///     # When using an inline table, can add a `reason` for why the method
    ///     # is disallowed.
    ///     { path = "std::vec::Vec::leak", reason = "no leaking memory" },
    /// ]
    /// ```
    ///
    /// ```rust,ignore
    /// // Example code where clippy issues a warning
    /// let xs = vec![1, 2, 3, 4];
    /// xs.leak(); // Vec::leak is disallowed in the config.
    /// // The diagnostic contains the message "no leaking memory".
    ///
    /// let _now = Instant::now(); // Instant::now is disallowed in the config.
    ///
    /// let _box = Box::new(3); // Box::new is disallowed in the config.
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// // Example code which does not raise clippy warning
    /// let mut xs = Vec::new(); // Vec::new is _not_ disallowed in the config.
    /// xs.push(123); // Vec::push is _not_ disallowed in the config.
    /// ```
    #[clippy::version = "1.49.0"]
    pub DISALLOWED_METHODS,
    style,
    "use of a disallowed method call"
}

#[derive(Clone, Debug)]
pub struct DisallowedMethods {
    conf_disallowed: Vec<conf::DisallowedMethod>,
    disallowed: DefIdMap<usize>,
}

impl DisallowedMethods {
    pub fn new(conf_disallowed: Vec<conf::DisallowedMethod>) -> Self {
        Self {
            conf_disallowed,
            disallowed: DefIdMap::default(),
        }
    }
}

impl_lint_pass!(DisallowedMethods => [DISALLOWED_METHODS]);

impl<'tcx> LateLintPass<'tcx> for DisallowedMethods {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        for (index, conf) in self.conf_disallowed.iter().enumerate() {
            let segs: Vec<_> = conf.path().split("::").collect();
            if let Res::Def(_, id) = clippy_utils::def_path_res(cx, &segs) {
                self.disallowed.insert(id, index);
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let def_id = match fn_def_id(cx, expr) {
            Some(def_id) => def_id,
            None => return,
        };
        let conf = match self.disallowed.get(&def_id) {
            Some(&index) => &self.conf_disallowed[index],
            None => return,
        };
        let msg = format!("use of a disallowed method `{}`", conf.path());
        span_lint_and_then(cx, DISALLOWED_METHODS, expr.span, &msg, |diag| {
            if let conf::DisallowedMethod::WithReason {
                reason: Some(reason), ..
            } = conf
            {
                diag.note(&format!("{} (from clippy.toml)", reason));
            }
        });
    }
}
