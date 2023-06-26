use crate::{context::LintContext, lints::ImplicitTransmuteTypesDiag, LateContext, LateLintPass};
use rustc_hir::{self as hir, ExprKind};
use rustc_span::symbol::sym;

declare_lint! {
    /// The `implicit_transmute_types` lint checks for calls to [`transmute`]
    /// without explicit type parameters (i.e. without turbofish syntax).
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(implicit_transmute_types)]
    ///
    /// #[repr(transparent)]
    /// struct CharWrapper {
    ///     _inner: char,
    /// }
    ///
    /// let wrapped = CharWrapper { _inner: 'a' };
    /// let transmuted = unsafe { core::mem::transmute(wrapped) };
    ///
    /// // This is sound now, but if it gets changed in the future to
    /// // something that expects a type other than `char`, the transmute
    /// // would infer it returns that type, which is likely unsound.
    /// // But because we have `deny(implicit_transmute_types)`, the
    /// // compiler will force us to come back and reassess whenever
    /// // the types change.
    /// let _ = char::is_lowercase(transmuted);
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In most cases Rust's type inference is helpful, however it can cause
    /// problems with [`transmute`]. `transmute` is wildly unsafe
    /// unless the types being transmuted are known to be compatible. As such,
    /// a seemingly innocent change in something's type can end up making a
    /// previously-valid `transmute` suddenly become unsound. Thus it is
    /// good practice to always be explicit about the types you expect to be
    /// transmuting between, so that the compiler will force you to
    /// reexamine the `transmute` if either type changes.
    ///
    /// If you have decided that you *do* want the types to be inferred,
    /// you can convey that explicitly:
    ///
    /// ```rust
    /// # use std::mem::transmute;
    /// # fn main() {
    /// #     unsafe {
    /// #         let foo: i32 = 123;
    /// #         let _: u32 =
    /// transmute::<_, _>(foo);
    /// #     }
    /// # }
    /// ```
    ///
    /// This lint is `allow` by default because it triggers on a lot of
    /// already-written existing code, including many instances in core
    /// libraries.
    ///
    /// [`transmute`]: https://doc.rust-lang.org/core/mem/fn.transmute.html
    IMPLICIT_TRANSMUTE_TYPES,
    Allow,
    "calling mem::transmute without explicit type parameters"
}

declare_lint_pass!(ImplicitTransmuteTypes => [IMPLICIT_TRANSMUTE_TYPES]);

impl<'tcx> LateLintPass<'tcx> for ImplicitTransmuteTypes {
    fn check_expr(&mut self, cx: &LateContext<'_>, e: &hir::Expr<'_>) {
        if let ExprKind::Call(func, _) = &e.kind
            && let ExprKind::Path(qpath) = &func.kind
            && let Some(def_id) = cx.qpath_res(qpath, func.hir_id).opt_def_id()
            && let Some(sym::transmute) = cx.tcx.get_diagnostic_name(def_id)
            && let hir::QPath::Resolved(_, path) = qpath
            && let Some(last_segment) = path.segments.last()
            && let None = last_segment.args
        {
            let suggestion_span = qpath.span().with_lo(qpath.span().hi());

            cx.emit_spanned_lint(
                IMPLICIT_TRANSMUTE_TYPES,
                e.span,
                ImplicitTransmuteTypesDiag { suggestion: suggestion_span }
            );
        }
    }
}
