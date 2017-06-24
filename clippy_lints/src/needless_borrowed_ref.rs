//! Checks for useless borrowed references in clojures.
//!
//! This lint is **warn** by default

use rustc::lint::*;
use rustc::hir::{MutImmutable, Pat, PatKind, BindingMode};
use rustc::ty;
use utils::{span_lint, in_macro};

/// **What it does:** Checks for useless borrowed references in clojures.
///
/// **Why is this bad?** TODO
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
///     let mut v = Vec::<String>::new();
///     let _ = v.iter_mut().filter(|&ref a| a.is_empty());
/// ```
/// It could just be |a| a.is_empty()
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
            // Pat is a pattern whose node
            // is a binding which "involves" a immutable reference...
            let PatKind::Binding(BindingMode::BindByRef(MutImmutable), ..) = pat.node,
            // Pattern's type is a reference. Get the type and mutability of referenced value (tam: TypeAndMut).
            let ty::TyRef(_, ref tam) = cx.tables.pat_ty(pat).sty,
            // This is an immutable reference.
            tam.mutbl == MutImmutable,
        ], {
            span_lint(cx, NEEDLESS_BORROWED_REFERENCE, pat.span, "this pattern takes a needless borrowed reference")
        }}
    }
}

