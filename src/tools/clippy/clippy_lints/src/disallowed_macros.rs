use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::macro_backtrace;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{Expr, ForeignItem, HirId, ImplItem, Item, Pat, Path, Stmt, TraitItem, Ty};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{ExpnId, Span};

use crate::utils::conf;

declare_clippy_lint! {
    /// ### What it does
    /// Denies the configured macros in clippy.toml
    ///
    /// Note: Even though this lint is warn-by-default, it will only trigger if
    /// macros are defined in the clippy.toml file.
    ///
    /// ### Why is this bad?
    /// Some macros are undesirable in certain contexts, and it's beneficial to
    /// lint for them as needed.
    ///
    /// ### Example
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-macros = [
    ///     # Can use a string as the path of the disallowed macro.
    ///     "std::print",
    ///     # Can also use an inline table with a `path` key.
    ///     { path = "std::println" },
    ///     # When using an inline table, can add a `reason` for why the macro
    ///     # is disallowed.
    ///     { path = "serde::Serialize", reason = "no serializing" },
    /// ]
    /// ```
    /// ```
    /// use serde::Serialize;
    ///
    /// // Example code where clippy issues a warning
    /// println!("warns");
    ///
    /// // The diagnostic will contain the message "no serializing"
    /// #[derive(Serialize)]
    /// struct Data {
    ///     name: String,
    ///     value: usize,
    /// }
    /// ```
    #[clippy::version = "1.66.0"]
    pub DISALLOWED_MACROS,
    style,
    "use of a disallowed macro"
}

pub struct DisallowedMacros {
    conf_disallowed: Vec<conf::DisallowedPath>,
    disallowed: DefIdMap<usize>,
    seen: FxHashSet<ExpnId>,
}

impl DisallowedMacros {
    pub fn new(conf_disallowed: Vec<conf::DisallowedPath>) -> Self {
        Self {
            conf_disallowed,
            disallowed: DefIdMap::default(),
            seen: FxHashSet::default(),
        }
    }

    fn check(&mut self, cx: &LateContext<'_>, span: Span) {
        if self.conf_disallowed.is_empty() {
            return;
        }

        for mac in macro_backtrace(span) {
            if !self.seen.insert(mac.expn) {
                return;
            }

            if let Some(&index) = self.disallowed.get(&mac.def_id) {
                let conf = &self.conf_disallowed[index];

                span_lint_and_then(
                    cx,
                    DISALLOWED_MACROS,
                    mac.span,
                    &format!("use of a disallowed macro `{}`", conf.path()),
                    |diag| {
                        if let Some(reason) = conf.reason() {
                            diag.note(reason);
                        }
                    },
                );
            }
        }
    }
}

impl_lint_pass!(DisallowedMacros => [DISALLOWED_MACROS]);

impl LateLintPass<'_> for DisallowedMacros {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        for (index, conf) in self.conf_disallowed.iter().enumerate() {
            let segs: Vec<_> = conf.path().split("::").collect();
            for id in clippy_utils::def_path_def_ids(cx, &segs) {
                self.disallowed.insert(id, index);
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        self.check(cx, expr.span);
    }

    fn check_stmt(&mut self, cx: &LateContext<'_>, stmt: &Stmt<'_>) {
        self.check(cx, stmt.span);
    }

    fn check_ty(&mut self, cx: &LateContext<'_>, ty: &Ty<'_>) {
        self.check(cx, ty.span);
    }

    fn check_pat(&mut self, cx: &LateContext<'_>, pat: &Pat<'_>) {
        self.check(cx, pat.span);
    }

    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        self.check(cx, item.span);
        self.check(cx, item.vis_span);
    }

    fn check_foreign_item(&mut self, cx: &LateContext<'_>, item: &ForeignItem<'_>) {
        self.check(cx, item.span);
        self.check(cx, item.vis_span);
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_>, item: &ImplItem<'_>) {
        self.check(cx, item.span);
        self.check(cx, item.vis_span);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_>, item: &TraitItem<'_>) {
        self.check(cx, item.span);
    }

    fn check_path(&mut self, cx: &LateContext<'_>, path: &Path<'_>, _: HirId) {
        self.check(cx, path.span);
    }
}
