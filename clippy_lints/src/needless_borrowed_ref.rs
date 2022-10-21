use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::Applicability;
use rustc_hir::{BindingAnnotation, Mutability, Node, Pat, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for bindings that needlessly destructure a reference and borrow the inner
    /// value with `&ref`.
    ///
    /// ### Why is this bad?
    /// This pattern has no effect in almost all cases.
    ///
    /// ### Example
    /// ```rust
    /// let mut v = Vec::<String>::new();
    /// v.iter_mut().filter(|&ref a| a.is_empty());
    ///
    /// if let &[ref first, ref second] = v.as_slice() {}
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let mut v = Vec::<String>::new();
    /// v.iter_mut().filter(|a| a.is_empty());
    ///
    /// if let [first, second] = v.as_slice() {}
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

        // Do not lint patterns that are part of an OR `|` pattern, the binding mode must match in all arms
        for (_, node) in cx.tcx.hir().parent_iter(pat.hir_id) {
            let Node::Pat(pat) = node else { break };

            if matches!(pat.kind, PatKind::Or(_)) {
                return;
            }
        }

        // Only lint immutable refs, because `&mut ref T` may be useful.
        let PatKind::Ref(sub_pat, Mutability::Not) = pat.kind else { return };

        match sub_pat.kind {
            // Check sub_pat got a `ref` keyword (excluding `ref mut`).
            PatKind::Binding(BindingAnnotation::REF, _, ident, None) => {
                span_lint_and_then(
                    cx,
                    NEEDLESS_BORROWED_REFERENCE,
                    pat.span,
                    "this pattern takes a reference on something that is being dereferenced",
                    |diag| {
                        // `&ref ident`
                        //  ^^^^^
                        let span = pat.span.until(ident.span);
                        diag.span_suggestion_verbose(
                            span,
                            "try removing the `&ref` part",
                            String::new(),
                            Applicability::MachineApplicable,
                        );
                    },
                );
            },
            // Slices where each element is `ref`: `&[ref a, ref b, ..., ref z]`
            PatKind::Slice(
                before,
                None
                | Some(Pat {
                    kind: PatKind::Wild, ..
                }),
                after,
            ) => {
                let mut suggestions = Vec::new();

                for element_pat in itertools::chain(before, after) {
                    if let PatKind::Binding(BindingAnnotation::REF, _, ident, None) = element_pat.kind {
                        // `&[..., ref ident, ...]`
                        //         ^^^^
                        let span = element_pat.span.until(ident.span);
                        suggestions.push((span, String::new()));
                    } else {
                        return;
                    }
                }

                if !suggestions.is_empty() {
                    span_lint_and_then(
                        cx,
                        NEEDLESS_BORROWED_REFERENCE,
                        pat.span,
                        "dereferencing a slice pattern where every element takes a reference",
                        |diag| {
                            // `&[...]`
                            //  ^
                            let span = pat.span.until(sub_pat.span);
                            suggestions.push((span, String::new()));

                            diag.multipart_suggestion(
                                "try removing the `&` and `ref` parts",
                                suggestions,
                                Applicability::MachineApplicable,
                            );
                        },
                    );
                }
            },
            _ => {},
        }
    }
}
