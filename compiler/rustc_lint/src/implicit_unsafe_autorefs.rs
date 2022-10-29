use crate::{LateContext, LateLintPass, LintContext};

use rustc_errors::Applicability;
use rustc_hir::{self as hir, Expr, ExprKind, UnOp};
use rustc_middle::ty::adjustment::{Adjust, AutoBorrow};

declare_lint! {
    /// The `implicit_unsafe_autorefs` lint checks for implicitly taken references to dereferences of raw pointers.
    ///
    /// ### Example
    ///
    /// ```rust
    /// unsafe fn fun(ptr: *mut [u8]) -> *mut [u8] {
    ///     addr_of_mut!((*ptr)[..16])
    ///     //                 ^^^^^^ this calls `IndexMut::index_mut(&mut ..., ..16)`,
    ///     //                        implicitly creating a reference
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// When working with raw pointers it's usually undesirable to create references,
    /// since they inflict a lot of safety requirement. Unfortunately, it's possible
    /// to take a reference to a dereferece of a raw pointer implitly, which inflicts
    /// the usual reference requirements without you even knowing that.
    /// 
    /// If you are sure, you can soundly take a reference, then you can take it explicitly:
    /// ```rust
    /// unsafe fn fun(ptr: *mut [u8]) -> *mut [u8] {
    ///     addr_of_mut!((&mut *ptr)[..16])
    /// }
    /// ```
    /// 
    /// Otherwise try to find an alternative way to achive your goals that work only with
    /// raw pointers:
    /// ```rust
    /// #![feature(slice_ptr_get)]
    /// 
    /// unsafe fn fun(ptr: *mut [u8]) -> *mut [u8] {
    ///     ptr.get_unchecked_mut(..16)
    /// }
    /// ```
    pub IMPLICIT_UNSAFE_AUTOREFS,
    Deny,
    "implicit reference to a dereference of a raw pointer"
}

declare_lint_pass!(ImplicitUnsafeAutorefs => [IMPLICIT_UNSAFE_AUTOREFS]);

impl<'tcx> LateLintPass<'tcx> for ImplicitUnsafeAutorefs {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let typeck = cx.typeck_results();
        let adjustments_table = typeck.adjustments();

        if let Some(adjustments) = adjustments_table.get(expr.hir_id) 
        && let [adjustment] = &**adjustments
        // An auto-borrow
        && let Adjust::Borrow(AutoBorrow::Ref(_, mutbl)) = adjustment.kind
        // ... of a deref
        && let ExprKind::Unary(UnOp::Deref, dereferenced) = expr.kind
        // ... of a raw pointer
        && typeck.expr_ty(dereferenced).is_unsafe_ptr()
        {
            let mutbl = hir::Mutability::prefix_str(&mutbl.into());
            
            let msg = "implicit auto-ref creates a reference to a dereference of a raw pointer";
            cx.struct_span_lint(IMPLICIT_UNSAFE_AUTOREFS, expr.span, msg, |lint| {
                lint
                    .note("creating a reference inflicts a lot of safety requirements")
                    .multipart_suggestion(
                        "if this reference is intentional, make it explicit", 
                        vec![
                            (expr.span.shrink_to_lo(), format!("(&{mutbl}")),
                            (expr.span.shrink_to_hi(), ")".to_owned())
                        ], 
                        Applicability::MaybeIncorrect
                    )
            })
        }
    }
}
