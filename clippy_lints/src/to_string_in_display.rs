use crate::utils::{match_def_path, match_trait_method, paths, qpath_res, span_lint};
use if_chain::if_chain;
use rustc_hir::def::Res;
use rustc_hir::{Expr, ExprKind, HirId, Impl, ImplItem, ImplItemKind, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// **What it does:** Checks for uses of `to_string()` in `Display` traits.
    ///
    /// **Why is this bad?** Usually `to_string` is implemented indirectly
    /// via `Display`. Hence using it while implementing `Display` would
    /// lead to infinite recursion.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
            if let ExprKind::MethodCall(ref path, _, args, _) = expr.kind;
            if path.ident.name == sym!(to_string);
            if match_trait_method(cx, expr, &paths::TO_STRING);
            if self.in_display_impl;
            if let ExprKind::Path(ref qpath) = args[0].kind;
            if let Res::Local(hir_id) = qpath_res(cx, qpath, args[0].hir_id);
            if let Some(self_hir_id) = self.self_hir_id;
            if hir_id == self_hir_id;
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
