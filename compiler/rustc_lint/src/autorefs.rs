use rustc_ast::{BorrowKind, UnOp};
use rustc_hir::{Expr, ExprKind, Mutability};
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow, OverloadedDeref};
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
    /// unsafe fn fun(ptr: *mut [u8]) -> *mut [u8] {
    ///     unsafe { &raw mut (*ptr)[..16] }
    ///     //                      ^^^^^^ this calls `IndexMut::index_mut(&mut ..., ..16)`,
    ///     //                             implicitly creating a reference
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// When working with raw pointers it's usually undesirable to create references,
    /// since they inflict additional safety requirements. Unfortunately, it's possible
    /// to take a reference to the dereference of a raw pointer implicitly, which inflicts
    /// the usual reference requirements.
    ///
    /// If you are sure that you can soundly take a reference, then you can take it explicitly:
    ///
    /// ```rust
    /// unsafe fn fun(ptr: *mut [u8]) -> *mut [u8] {
    ///     unsafe { &raw mut (&mut *ptr)[..16] }
    /// }
    /// ```
    ///
    /// Otherwise try to find an alternative way to achive your goals using only raw pointers:
    ///
    /// ```rust
    /// use std::ptr;
    ///
    /// fn fun(ptr: *mut [u8]) -> *mut [u8] {
    ///     ptr::slice_from_raw_parts_mut(ptr.cast(), 16)
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
        // <https://github.com/rust-lang/rust/pull/103735#issuecomment-1370420305>

        // 5. Either of the following:
        //   a. A deref followed by any non-deref place projection (that intermediate
        //      deref will typically be auto-inserted).
        //   b. A method call annotated with `#[rustc_no_implicit_refs]`.
        //   c. A deref followed by a `&raw const` or `&raw mut`.
        let mut is_coming_from_deref = false;
        let inner = match expr.kind {
            ExprKind::AddrOf(BorrowKind::Raw, _, inner) => match inner.kind {
                ExprKind::Unary(UnOp::Deref, inner) => {
                    is_coming_from_deref = true;
                    inner
                }
                _ => return,
            },
            ExprKind::Index(base, _, _) => base,
            ExprKind::MethodCall(_, inner, _, _) => {
                // PERF: Checking of `#[rustc_no_implicit_refs]` is deferred below
                // because checking for attribute is a bit costly.
                inner
            }
            ExprKind::Field(inner, _) => inner,
            _ => return,
        };

        let typeck = cx.typeck_results();
        let adjustments_table = typeck.adjustments();

        if let Some(adjustments) = adjustments_table.get(inner.hir_id)
            // 4. Any number of automatically inserted deref/derefmut calls.
            && let adjustments = peel_derefs_adjustments(&**adjustments)
            // 3. An automatically inserted reference (might come from a deref).
            && let [adjustment] = adjustments
            && let Some(borrow_mutbl) = has_implicit_borrow(adjustment)
            && let ExprKind::Unary(UnOp::Deref, dereferenced) =
                // 2. Any number of place projections.
                peel_place_mappers(inner).kind
            // 1. Deref of a raw pointer.
            && typeck.expr_ty(dereferenced).is_raw_ptr()
            // PERF: 5. b. A method call annotated with `#[rustc_no_implicit_refs]`
            && match expr.kind {
                ExprKind::MethodCall(..) => matches!(
                    cx.typeck_results().type_dependent_def_id(expr.hir_id),
                    Some(def_id) if cx.tcx.has_attr(def_id, sym::rustc_no_implicit_autorefs)
                ),
                _ => true,
            }
        {
            cx.emit_span_lint(
                DANGEROUS_IMPLICIT_AUTOREFS,
                expr.span.source_callsite(),
                ImplicitUnsafeAutorefsDiag {
                    suggestion: ImplicitUnsafeAutorefsSuggestion {
                        mutbl: borrow_mutbl.ref_prefix_str(),
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
fn peel_place_mappers<'tcx>(mut expr: &'tcx Expr<'tcx>) -> &'tcx Expr<'tcx> {
    loop {
        match expr.kind {
            ExprKind::Index(base, _idx, _) => expr = &base,
            ExprKind::Field(e, _) => expr = &e,
            _ => break expr,
        }
    }
}

/// Peel derefs adjustments until the last last element.
fn peel_derefs_adjustments<'a>(mut adjs: &'a [Adjustment<'a>]) -> &'a [Adjustment<'a>] {
    while let [Adjustment { kind: Adjust::Deref(_), .. }, end @ ..] = adjs
        && !end.is_empty()
    {
        adjs = end;
    }
    adjs
}

/// Test if some adjustment has some implicit borrow.
///
/// Returns `Some(mutability)` if the argument adjustment has implicit borrow in it.
fn has_implicit_borrow(Adjustment { kind, .. }: &Adjustment<'_>) -> Option<Mutability> {
    match kind {
        &Adjust::Deref(Some(OverloadedDeref { mutbl, .. })) => Some(mutbl),
        &Adjust::Borrow(AutoBorrow::Ref(mutbl)) => Some(mutbl.into()),
        Adjust::NeverToAny
        | Adjust::Pointer(..)
        | Adjust::ReborrowPin(..)
        | Adjust::Deref(None)
        | Adjust::Borrow(AutoBorrow::RawPtr(..)) => None,
    }
}
