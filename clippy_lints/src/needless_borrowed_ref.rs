//! Checks for useless borrowed references.
//!
//! This lint is **warn** by default

use rustc::lint::*;
<<<<<<< HEAD
use rustc::hir::{MutImmutable, Pat, PatKind, BindingAnnotation};
=======
use rustc::hir::{MutImmutable, Pat, PatKind};
>>>>>>> e30bf721... Improve needless_borrowed_ref and add suggestion to it.
use rustc::ty;
use utils::{span_lint_and_then, in_macro, snippet};
use syntax_pos::{Span, BytePos};

/// **What it does:** Checks for useless borrowed references.
///
/// **Why is this bad?** It is completely useless and make the code look more
/// complex than it
/// actually is.
///
/// **Known problems:** None.
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
            // Pat is a pattern whose node
            // is a binding which "involves" an immutable reference...
            let PatKind::Binding(BindingAnnotation::Ref, ..) = pat.node,
            // Pattern's type is a reference. Get the type and mutability of referenced value (tam: TypeAndMut).
            let ty::TyRef(_, ref tam) = cx.tables.pat_ty(pat).sty,
            // Only lint immutable refs, because `&mut ref T` may be useful.
            let PatKind::Ref(ref sub_pat, MutImmutable) = pat.node,

            // Check sub_pat got a 'ref' keyword.
            let ty::TyRef(_, _) = cx.tables.pat_ty(sub_pat).sty,
        ], {
            let part_to_keep = Span{ lo: pat.span.lo + BytePos(5), hi: pat.span.hi, ctxt: pat.span.ctxt };
            span_lint_and_then(cx, NEEDLESS_BORROWED_REFERENCE, pat.span,
                               "this pattern takes a reference on something that is being de-referenced",
                               |db| {
                                   let hint = snippet(cx, part_to_keep, "..").into_owned();
                                   db.span_suggestion(pat.span, "try removing the `&ref` part and just keep", hint);
                               });
        }}
    }
}
