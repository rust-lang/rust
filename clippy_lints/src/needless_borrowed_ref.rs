//! Checks for useless borrowed references.
//!
//! This lint is **warn** by default

use crate::utils::{in_macro_or_desugar, snippet, span_lint_and_then};
use if_chain::if_chain;
use rustc::hir::{BindingAnnotation, MutImmutable, Pat, PatKind};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;

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
    fn check_pat(&mut self, cx: &LateContext<'a, 'tcx>, pat: &'tcx Pat) {
        if in_macro_or_desugar(pat.span) {
            // OK, simple enough, lints doesn't check in macro.
            return;
        }

        if_chain! {
            // Only lint immutable refs, because `&mut ref T` may be useful.
            if let PatKind::Ref(ref sub_pat, MutImmutable) = pat.node;

            // Check sub_pat got a `ref` keyword (excluding `ref mut`).
            if let PatKind::Binding(BindingAnnotation::Ref, .., spanned_name, _) = sub_pat.node;
            then {
                span_lint_and_then(cx, NEEDLESS_BORROWED_REFERENCE, pat.span,
                                   "this pattern takes a reference on something that is being de-referenced",
                                   |db| {
                                       let hint = snippet(cx, spanned_name.span, "..").into_owned();
                                       db.span_suggestion(
                                           pat.span,
                                           "try removing the `&ref` part and just keep",
                                           hint,
                                           Applicability::MachineApplicable, // snippet
                                       );
                                   });
            }
        }
    }
}
