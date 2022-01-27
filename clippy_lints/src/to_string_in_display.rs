use clippy_utils::diagnostics::span_lint;
use clippy_utils::{is_diag_trait_item, match_def_path, path_to_local_id, paths};
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, HirId, Impl, ImplItem, ImplItemKind, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for uses of `to_string()` in `Display` traits.
    ///
    /// ### Why is this bad?
    /// Usually `to_string` is implemented indirectly
    /// via `Display`. Hence using it while implementing `Display` would
    /// lead to infinite recursion.
    ///
    /// ### Example
    ///
    /// ```rust
    /// use std::fmt;
    ///
    /// struct Structure(i32);
    /// impl fmt::Display for Structure {
    ///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    ///         write!(f, "{}", self.to_string())
    ///     }
    /// }
    ///
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::fmt;
    ///
    /// struct Structure(i32);
    /// impl fmt::Display for Structure {
    ///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    ///         write!(f, "{}", self.0)
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.48.0"]
    pub TO_STRING_IN_DISPLAY,
    correctness,
    "`to_string` method used while implementing `Display` trait"
}

#[derive(Default)]
pub struct ToStringInDisplay {
    in_display_impl: bool,
    self_hir_id: Option<HirId>,
}

impl ToStringInDisplay {
    pub fn new() -> Self {
        Self {
            in_display_impl: false,
            self_hir_id: None,
        }
    }
}

impl_lint_pass!(ToStringInDisplay => [TO_STRING_IN_DISPLAY]);

impl LateLintPass<'_> for ToStringInDisplay {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if is_display_impl(cx, item) {
            self.in_display_impl = true;
        }
    }

    fn check_item_post(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if is_display_impl(cx, item) {
            self.in_display_impl = false;
            self.self_hir_id = None;
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_>, impl_item: &ImplItem<'_>) {
        if_chain! {
            if self.in_display_impl;
            if let ImplItemKind::Fn(.., body_id) = &impl_item.kind;
            let body = cx.tcx.hir().body(*body_id);
            if !body.params.is_empty();
            then {
                let self_param = &body.params[0];
                self.self_hir_id = Some(self_param.pat.hir_id);
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if self.in_display_impl;
            if let Some(self_hir_id) = self.self_hir_id;
            if let ExprKind::MethodCall(path, [ref self_arg, ..], _) = expr.kind;
            if path.ident.name == sym!(to_string);
            if let Some(expr_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
            if is_diag_trait_item(cx, expr_def_id, sym::ToString);
            if path_to_local_id(self_arg, self_hir_id);
            then {
                span_lint(
                    cx,
                    TO_STRING_IN_DISPLAY,
                    expr.span,
                    "using `to_string` in `fmt::Display` implementation might lead to infinite recursion",
                );
            }
        }
    }
}

fn is_display_impl(cx: &LateContext<'_>, item: &Item<'_>) -> bool {
    if_chain! {
        if let ItemKind::Impl(Impl { of_trait: Some(trait_ref), .. }) = &item.kind;
        if let Some(did) = trait_ref.trait_def_id();
        then {
            match_def_path(cx, did, &paths::DISPLAY_TRAIT)
        } else {
            false
        }
    }
}
