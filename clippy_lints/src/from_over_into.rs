use crate::utils::paths::INTO;
use crate::utils::{match_def_path, span_lint_and_help};
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Searches for implementations of the `Into<..>` trait and suggests to implement `From<..>` instead.
    ///
    /// **Why is this bad?** According the std docs implementing `From<..>` is preferred since it gives you `Into<..>` for free where the reverse isn't true.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// struct StringWrapper(String);
    ///
    /// impl Into<StringWrapper> for String {
    ///     fn into(self) -> StringWrapper {
    ///         StringWrapper(self)
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// struct StringWrapper(String);
    ///
    /// impl From<String> for StringWrapper {
    ///     fn from(s: String) -> StringWrapper {
    ///         StringWrapper(s)
    ///     }
    /// }
    /// ```
    pub FROM_OVER_INTO,
    style,
    "Warns on implementations of `Into<..>` to use `From<..>`"
}

declare_lint_pass!(FromOverInto => [FROM_OVER_INTO]);

impl LateLintPass<'_> for FromOverInto {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        let impl_def_id = cx.tcx.hir().local_def_id(item.hir_id);
        if_chain! {
            if let hir::ItemKind::Impl{ .. } = &item.kind;
            if let Some(impl_trait_ref) = cx.tcx.impl_trait_ref(impl_def_id);
            if match_def_path(cx, impl_trait_ref.def_id, &INTO);

            then {
                span_lint_and_help(
                    cx,
                    FROM_OVER_INTO,
                    item.span,
                    "An implementation of From is preferred since it gives you Into<..> for free where the reverse isn't true.",
                    None,
                    "consider implement From instead",
                );
            }
        }
    }
}
