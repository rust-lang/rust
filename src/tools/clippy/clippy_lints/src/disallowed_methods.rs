use clippy_config::Conf;
use clippy_config::types::create_disallowed_map;
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;

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

pub struct DisallowedMethods {
    disallowed: DefIdMap<(&'static str, Option<&'static str>)>,
}

impl DisallowedMethods {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        Self {
            disallowed: create_disallowed_map(tcx, &conf.disallowed_methods),
        }
    }
}

impl_lint_pass!(DisallowedMethods => [DISALLOWED_METHODS]);

impl<'tcx> LateLintPass<'tcx> for DisallowedMethods {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let (id, span) = match &expr.kind {
            ExprKind::Path(path)
                if let Res::Def(DefKind::Fn | DefKind::Ctor(_, CtorKind::Fn) | DefKind::AssocFn, id) =
                    cx.qpath_res(path, expr.hir_id) =>
            {
                (id, expr.span)
            },
            ExprKind::MethodCall(name, ..) if let Some(id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) => {
                (id, name.ident.span)
            },
            _ => return,
        };
        if let Some(&(path, reason)) = self.disallowed.get(&id) {
            span_lint_and_then(
                cx,
                DISALLOWED_METHODS,
                span,
                format!("use of a disallowed method `{path}`"),
                |diag| {
                    if let Some(reason) = reason {
                        diag.note(reason);
                    }
                },
            );
        }
    }
}
