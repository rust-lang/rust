use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{meets_msrv, msrvs};
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Searches for implementations of the `Into<..>` trait and suggests to implement `From<..>` instead.
    ///
    /// ### Why is this bad?
    /// According the std docs implementing `From<..>` is preferred since it gives you `Into<..>` for free where the reverse isn't true.
    ///
    /// ### Example
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

pub struct FromOverInto {
    msrv: Option<RustcVersion>,
}

impl FromOverInto {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        FromOverInto { msrv }
    }
}

impl_lint_pass!(FromOverInto => [FROM_OVER_INTO]);

impl LateLintPass<'_> for FromOverInto {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        if !meets_msrv(self.msrv.as_ref(), &msrvs::RE_REBALANCING_COHERENCE) {
            return;
        }

        if_chain! {
            if let hir::ItemKind::Impl{ .. } = &item.kind;
            if let Some(impl_trait_ref) = cx.tcx.impl_trait_ref(item.def_id);
            if cx.tcx.is_diagnostic_item(sym::Into, impl_trait_ref.def_id);

            then {
                span_lint_and_help(
                    cx,
                    FROM_OVER_INTO,
                    cx.tcx.sess.source_map().guess_head_span(item.span),
                    "an implementation of `From` is preferred since it gives you `Into<_>` for free where the reverse isn't true",
                    None,
                    &format!("consider to implement `From<{}>` instead", impl_trait_ref.self_ty()),
                );
            }
        }
    }

    extract_msrv_attr!(LateContext);
}
