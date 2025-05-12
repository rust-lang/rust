use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::Applicability;
use rustc_hir::{BindingMode, Mutability, Node, Pat, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for bindings that needlessly destructure a reference and borrow the inner
    /// value with `&ref`.
    ///
    /// ### Why is this bad?
    /// This pattern has no effect in almost all cases.
    ///
    /// ### Example
    /// ```no_run
    /// let mut v = Vec::<String>::new();
    /// v.iter_mut().filter(|&ref a| a.is_empty());
    ///
    /// if let &[ref first, ref second] = v.as_slice() {}
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    fn check_pat(&mut self, cx: &LateContext<'tcx>, ref_pat: &'tcx Pat<'_>) {
        if let PatKind::Ref(pat, Mutability::Not) = ref_pat.kind
            && !ref_pat.span.from_expansion()
            && cx
                .tcx
                .hir_parent_iter(ref_pat.hir_id)
                .map_while(|(_, parent)| if let Node::Pat(pat) = parent { Some(pat) } else { None })
                // Do not lint patterns that are part of an OR `|` pattern, the binding mode must match in all arms
                .all(|pat| !matches!(pat.kind, PatKind::Or(_)))
        {
            match pat.kind {
                // Check sub_pat got a `ref` keyword (excluding `ref mut`).
                PatKind::Binding(BindingMode::REF, _, ident, None) => {
                    span_lint_and_then(
                        cx,
                        NEEDLESS_BORROWED_REFERENCE,
                        ref_pat.span,
                        "this pattern takes a reference on something that is being dereferenced",
                        |diag| {
                            // `&ref ident`
                            //  ^^^^^
                            let span = ref_pat.span.until(ident.span);
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
                    check_subpatterns(
                        cx,
                        "dereferencing a slice pattern where every element takes a reference",
                        ref_pat,
                        pat,
                        itertools::chain(before, after),
                    );
                },
                PatKind::Tuple(subpatterns, _) | PatKind::TupleStruct(_, subpatterns, _) => {
                    check_subpatterns(
                        cx,
                        "dereferencing a tuple pattern where every element takes a reference",
                        ref_pat,
                        pat,
                        subpatterns,
                    );
                },
                PatKind::Struct(_, fields, _) => {
                    check_subpatterns(
                        cx,
                        "dereferencing a struct pattern where every field's pattern takes a reference",
                        ref_pat,
                        pat,
                        fields.iter().map(|field| field.pat),
                    );
                },
                _ => {},
            }
        }
    }
}

fn check_subpatterns<'tcx>(
    cx: &LateContext<'tcx>,
    message: &'static str,
    ref_pat: &Pat<'_>,
    pat: &Pat<'_>,
    subpatterns: impl IntoIterator<Item = &'tcx Pat<'tcx>>,
) {
    let mut suggestions = Vec::new();

    for subpattern in subpatterns {
        match subpattern.kind {
            PatKind::Binding(BindingMode::REF, _, ident, None) => {
                // `ref ident`
                //  ^^^^
                let span = subpattern.span.until(ident.span);
                suggestions.push((span, String::new()));
            },
            PatKind::Wild => {},
            _ => return,
        }
    }

    if !suggestions.is_empty() {
        span_lint_and_then(cx, NEEDLESS_BORROWED_REFERENCE, ref_pat.span, message, |diag| {
            // `&pat`
            //  ^
            let span = ref_pat.span.until(pat.span);
            suggestions.push((span, String::new()));

            diag.multipart_suggestion(
                "try removing the `&` and `ref` parts",
                suggestions,
                Applicability::MachineApplicable,
            );
        });
    }
}
