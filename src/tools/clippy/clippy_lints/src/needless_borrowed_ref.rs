//! Checks for useless borrowed references.
//!
//! This lint is **warn** by default

use crate::utils::{snippet_with_applicability, span_lint_and_then};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BindingAnnotation, Mutability, Node, Pat, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for useless borrowed references.
    ///
    /// **Why is this bad?** It is mostly useless and make the code look more
    /// complex than it
    /// actually is.
    ///
    /// **Known problems:** It seems that the `&ref` pattern is sometimes useful.
    /// For instance in the following snippet:
    /// ```rust,ignore
    /// enum Animal {
    ///     Cat(u64),
    ///     Dog(u64),
    /// }
    ///
    /// fn foo(a: &Animal, b: &Animal) {
    ///     match (a, b) {
    ///         (&Animal::Cat(v), k) | (k, &Animal::Cat(v)) => (), // lifetime mismatch error
    ///         (&Animal::Dog(ref c), &Animal::Dog(_)) => ()
    ///     }
    /// }
    /// ```
    /// There is a lifetime mismatch error for `k` (indeed a and b have distinct
    /// lifetime).
    /// This can be fixed by using the `&ref` pattern.
    /// However, the code can also be fixed by much cleaner ways
    ///
    /// **Example:**
    /// ```rust
    /// let mut v = Vec::<String>::new();
    /// let _ = v.iter_mut().filter(|&ref a| a.is_empty());
    /// ```
    /// This closure takes a reference on something that has been matched as a
    /// reference and
    /// de-referenced.
    /// As such, it could just be |a| a.is_empty()
    pub NEEDLESS_BORROWED_REFERENCE,
    complexity,
    "taking a needless borrowed reference"
}

declare_lint_pass!(NeedlessBorrowedRef => [NEEDLESS_BORROWED_REFERENCE]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NeedlessBorrowedRef {
    fn check_pat(&mut self, cx: &LateContext<'a, 'tcx>, pat: &'tcx Pat<'_>) {
        if pat.span.from_expansion() {
            // OK, simple enough, lints doesn't check in macro.
            return;
        }

        if_chain! {
            // Only lint immutable refs, because `&mut ref T` may be useful.
            if let PatKind::Ref(ref sub_pat, Mutability::Not) = pat.kind;

            // Check sub_pat got a `ref` keyword (excluding `ref mut`).
            if let PatKind::Binding(BindingAnnotation::Ref, .., spanned_name, _) = sub_pat.kind;
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
