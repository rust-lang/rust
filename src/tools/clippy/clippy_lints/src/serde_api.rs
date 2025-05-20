use clippy_utils::diagnostics::span_lint;
use clippy_utils::paths;
use rustc_hir::{Impl, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for misuses of the serde API.
    ///
    /// ### Why is this bad?
    /// Serde is very finicky about how its API should be
    /// used, but the type system can't be used to enforce it (yet?).
    ///
    /// ### Example
    /// Implementing `Visitor::visit_string` but not
    /// `Visitor::visit_str`.
    #[clippy::version = "pre 1.29.0"]
    pub SERDE_API_MISUSE,
    correctness,
    "various things that will negatively affect your serde experience"
}

declare_lint_pass!(SerdeApi => [SERDE_API_MISUSE]);

impl<'tcx> LateLintPass<'tcx> for SerdeApi {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Impl(Impl {
            of_trait: Some(trait_ref),
            items,
            ..
        }) = item.kind
        {
            let did = trait_ref.path.res.def_id();
            if paths::SERDE_DE_VISITOR.matches(cx, did) {
                let mut seen_str = None;
                let mut seen_string = None;
                for item in *items {
                    match item.ident.as_str() {
                        "visit_str" => seen_str = Some(item.span),
                        "visit_string" => seen_string = Some(item.span),
                        _ => {},
                    }
                }
                if let Some(span) = seen_string
                    && seen_str.is_none()
                {
                    span_lint(
                        cx,
                        SERDE_API_MISUSE,
                        span,
                        "you should not implement `visit_string` without also implementing `visit_str`",
                    );
                }
            }
        }
    }
}
