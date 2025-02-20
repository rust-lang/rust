use rustc_ast::{BorrowKind, UnOp};
use rustc_hir::{Expr, ExprKind, Mutability};
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow, OverloadedDeref};
use rustc_middle::ty::{TyCtxt, TypeckResults};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::sym;

use crate::lints::{ImplicitUnsafeAutorefsDiag, ImplicitUnsafeAutorefsSuggestion};
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `dangerous_implicit_autorefs` lint checks for implicitly taken references
    /// to dereferences of raw pointers.
    ///
    /// ### Example
    ///
    /// ```rust
    /// use std::ptr::addr_of_mut;
    ///
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
    /// to take a reference to a dereference of a raw pointer implicitly, which inflicts
    /// the usual reference requirements without you even knowing that.
    ///
    /// If you are sure, you can soundly take a reference, then you can take it explicitly:
    /// ```rust
    /// # use std::ptr::addr_of_mut;
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
    pub DANGEROUS_IMPLICIT_AUTOREFS,
    Warn,
    "implicit reference to a dereference of a raw pointer",
    report_in_external_macro
}

declare_lint_pass!(ImplicitAutorefs => [DANGEROUS_IMPLICIT_AUTOREFS]);

impl<'tcx> LateLintPass<'tcx> for ImplicitAutorefs {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // This logic has mostly been taken from
        // https://github.com/rust-lang/rust/pull/103735#issuecomment-1370420305

        // 4. Either of the following:
        //   a. A deref followed by any non-deref place projection (that intermediate
        //   deref will typically be auto-inserted)
        //   b. A method call annotated with `#[rustc_no_implicit_refs]`.
        //   c. A deref followed by a `addr_of!` or `addr_of_mut!`.
        let mut is_coming_from_deref = false;
        let inner = match expr.kind {
            ExprKind::AddrOf(BorrowKind::Raw, _, inner) => match inner.kind {
                ExprKind::Unary(UnOp::Deref, inner) => {
                    is_coming_from_deref = true;
                    inner
                }
                _ => return,
            },
            ExprKind::Index(base, _idx, _) => base,
            ExprKind::MethodCall(_, inner, _, _)
                if let Some(def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
                    && cx.tcx.has_attr(def_id, sym::rustc_no_implicit_autorefs) =>
            {
                inner
            }
            ExprKind::Call(path, [expr, ..])
                if let ExprKind::Path(ref qpath) = path.kind
                    && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
                    && cx.tcx.has_attr(def_id, sym::rustc_no_implicit_autorefs) =>
            {
                expr
            }
            ExprKind::Field(inner, _) => {
                let typeck = cx.typeck_results();
                let adjustments_table = typeck.adjustments();
                if let Some(adjustments) = adjustments_table.get(inner.hir_id)
                    && let [adjustment] = &**adjustments
                    && let &Adjust::Deref(Some(OverloadedDeref { .. })) = &adjustment.kind
                {
                    inner
                } else {
                    return;
                }
            }
            _ => return,
        };

        let typeck = cx.typeck_results();
        let adjustments_table = typeck.adjustments();

        if let Some(adjustments) = adjustments_table.get(inner.hir_id)
            && let [adjustment] = &**adjustments
            // 3. An automatically inserted reference.
            && let Some((mutbl, _implicit_borrow)) = has_implicit_borrow(adjustment)
            && let ExprKind::Unary(UnOp::Deref, dereferenced) =
                // 2. Any number of place projections
                peel_place_mappers(cx.tcx, typeck, inner).kind
            // 1. Deref of a raw pointer
            && typeck.expr_ty(dereferenced).is_unsafe_ptr()
        {
            cx.emit_span_lint(
                DANGEROUS_IMPLICIT_AUTOREFS,
                expr.span.source_callsite(),
                ImplicitUnsafeAutorefsDiag {
                    suggestion: ImplicitUnsafeAutorefsSuggestion {
                        mutbl: mutbl.ref_prefix_str(),
                        deref: if is_coming_from_deref { "*" } else { "" },
                        start_span: inner.span.shrink_to_lo(),
                        end_span: inner.span.shrink_to_hi(),
                    },
                },
            )
        }
    }
}

/// Peels expressions from `expr` that can map a place.
fn peel_place_mappers<'tcx>(
    _tcx: TyCtxt<'tcx>,
    _typeck: &TypeckResults<'tcx>,
    mut expr: &'tcx Expr<'tcx>,
) -> &'tcx Expr<'tcx> {
    loop {
        match expr.kind {
            ExprKind::Index(base, _idx, _) => {
                expr = &base;
            }
            ExprKind::Field(e, _) => expr = &e,
            _ => break expr,
        }
    }
}

enum ImplicitBorrowKind {
    Deref,
    Borrow,
}

/// Test if some adjustment has some implicit borrow
///
/// Returns `Some(mutability)` if the argument adjustment has implicit borrow in it.
fn has_implicit_borrow(
    Adjustment { kind, .. }: &Adjustment<'_>,
) -> Option<(Mutability, ImplicitBorrowKind)> {
    match kind {
        &Adjust::Deref(Some(OverloadedDeref { mutbl, .. })) => {
            Some((mutbl, ImplicitBorrowKind::Deref))
        }
        &Adjust::Borrow(AutoBorrow::Ref(_, mutbl)) => {
            Some((mutbl.into(), ImplicitBorrowKind::Borrow))
        }
        _ => None,
    }
}
