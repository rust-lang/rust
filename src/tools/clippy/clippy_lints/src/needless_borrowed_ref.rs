use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BindingAnnotation, Mutability, Node, Pat, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for bindings that destructure a reference and borrow the inner
    /// value with `&ref`.
    ///
    /// ### Why is this bad?
    /// This pattern has no effect in almost all cases.
    ///
    /// ### Known problems
    /// In some cases, `&ref` is needed to avoid a lifetime mismatch error.
    /// Example:
    /// ```rust
    /// fn foo(a: &Option<String>, b: &Option<String>) {
    ///     match (a, b) {
    ///         (None, &ref c) | (&ref c, None) => (),
    ///         (&Some(ref c), _) => (),
    ///     };
    /// }
    /// ```
    ///
    /// ### Example
    /// ```rust
    /// let mut v = Vec::<String>::new();
    /// # #[allow(unused)]
    /// v.iter_mut().filter(|&ref a| a.is_empty());
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let mut v = Vec::<String>::new();
    /// # #[allow(unused)]
    /// v.iter_mut().filter(|a| a.is_empty());
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEEDLESS_BORROWED_REFERENCE,
    complexity,
    "destructuring a reference and borrowing the inner value"
}

declare_lint_pass!(NeedlessBorrowedRef => [NEEDLESS_BORROWED_REFERENCE]);

impl<'tcx> LateLintPass<'tcx> for NeedlessBorrowedRef {
    fn check_pat(&mut self, cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>) {
        if pat.span.from_expansion() {
            // OK, simple enough, lints doesn't check in macro.
            return;
        }

        if_chain! {
            // Only lint immutable refs, because `&mut ref T` may be useful.
            if let PatKind::Ref(sub_pat, Mutability::Not) = pat.kind;

            // Check sub_pat got a `ref` keyword (excluding `ref mut`).
            if let PatKind::Binding(BindingAnnotation::REF, .., spanned_name, _) = sub_pat.kind;
            let parent_id = cx.tcx.hir().get_parent_node(pat.hir_id);
            if let Some(parent_node) = cx.tcx.hir().find(parent_id);
            then {
                // do not recurse within patterns, as they may have other references
                // XXXManishearth we can relax this constraint if we only check patterns
                // with a single ref pattern inside them
                if let Node::Pat(_) = parent_node {
                    return;
                }
                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_then(cx, NEEDLESS_BORROWED_REFERENCE, pat.span,
                                   "this pattern takes a reference on something that is being de-referenced",
                                   |diag| {
                                       let hint = snippet_with_applicability(cx, spanned_name.span, "..", &mut applicability).into_owned();
                                       diag.span_suggestion(
                                           pat.span,
                                           "try removing the `&ref` part and just keep",
                                           hint,
                                           applicability,
                                       );
                                   });
            }
        }
    }
}
