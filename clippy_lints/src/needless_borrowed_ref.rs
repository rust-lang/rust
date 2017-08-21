//! Checks for useless borrowed references.
//!
//! This lint is **warn** by default

use rustc::lint::*;
use rustc::hir::{MutImmutable, Pat, PatKind, BindingAnnotation};
use utils::{span_lint_and_then, in_macro, snippet};

/// **What it does:** Checks for useless borrowed references.
///
/// **Why is this bad?** It is mostly useless and make the code look more complex than it
/// actually is.
///
/// **Known problems:** It seems that the `&ref` pattern is sometimes useful.
/// For instance in the following snippet:
/// ```rust
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
/// There is a lifetime mismatch error for `k` (indeed a and b have distinct lifetime).
/// This can be fixed by using the `&ref` pattern.
/// However, the code can also be fixed by much cleaner ways
///
/// **Example:**
/// ```rust
///     let mut v = Vec::<String>::new();
///     let _ = v.iter_mut().filter(|&ref a| a.is_empty());
/// ```
/// This clojure takes a reference on something that has been matched as a
/// reference and
/// de-referenced.
/// As such, it could just be |a| a.is_empty()
declare_lint! {
    pub NEEDLESS_BORROWED_REFERENCE,
    Warn,
    "taking a needless borrowed reference"
}

#[derive(Copy, Clone)]
pub struct NeedlessBorrowedRef;

impl LintPass for NeedlessBorrowedRef {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_BORROWED_REFERENCE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NeedlessBorrowedRef {
    fn check_pat(&mut self, cx: &LateContext<'a, 'tcx>, pat: &'tcx Pat) {
        if in_macro(pat.span) {
            // OK, simple enough, lints doesn't check in macro.
            return;
        }

        if_let_chain! {[
            // Only lint immutable refs, because `&mut ref T` may be useful.
            let PatKind::Ref(ref sub_pat, MutImmutable) = pat.node,

            // Check sub_pat got a `ref` keyword (excluding `ref mut`).
            let PatKind::Binding(BindingAnnotation::Ref, _, spanned_name, ..) = sub_pat.node,
        ], {
            span_lint_and_then(cx, NEEDLESS_BORROWED_REFERENCE, pat.span,
                               "this pattern takes a reference on something that is being de-referenced",
                               |db| {
                                   let hint = snippet(cx, spanned_name.span, "..").into_owned();
                                   db.span_suggestion(pat.span, "try removing the `&ref` part and just keep", hint);
                               });
        }}
    }
}

