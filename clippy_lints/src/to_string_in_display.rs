use crate::utils::{match_def_path, match_trait_method, paths, span_lint};
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, Item, ItemKind};
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
    "to_string method used while implementing Display trait"
}

#[derive(Default)]
pub struct ToStringInDisplay {
    in_display_impl: bool,
}

impl ToStringInDisplay {
    pub fn new() -> Self {
        Self { in_display_impl: false }
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
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if let ExprKind::MethodCall(ref path, _, _, _) = expr.kind;
            if path.ident.name == sym!(to_string);
            if match_trait_method(cx, expr, &paths::TO_STRING);
            if self.in_display_impl;

            then {
                span_lint(
                    cx,
                    TO_STRING_IN_DISPLAY,
                    expr.span,
                    "Using to_string in fmt::Display implementation might lead to infinite recursion",
                );
            }
        }
    }
}

fn is_display_impl(cx: &LateContext<'_>, item: &Item<'_>) -> bool {
    if_chain! {
        if let ItemKind::Impl { of_trait: Some(trait_ref), .. } = &item.kind;
        if let Some(did) = trait_ref.trait_def_id();
        then {
            match_def_path(cx, did, &paths::DISPLAY_TRAIT)
        } else {
            false
        }
    }
}
