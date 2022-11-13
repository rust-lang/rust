use crate::{LateContext, LateLintPass, LintContext};

use rustc_errors::Applicability;
use rustc_hir::{self as hir, Expr, ExprKind, Mutability, UnOp};
use rustc_middle::ty::{
    adjustment::{Adjust, Adjustment, AutoBorrow, OverloadedDeref},
    TyCtxt, TypeckResults,
};

declare_lint! {
    /// The `implicit_unsafe_autorefs` lint checks for implicitly taken references to dereferences of raw pointers.
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
    pub IMPLICIT_UNSAFE_AUTOREFS,
    Warn,
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
        && let Some((mutbl, implicit_borrow)) = has_implicit_borrow(adjustment)
        // ... of a place derived from a deref
        && let ExprKind::Unary(UnOp::Deref, dereferenced) = peel_place_mappers(cx.tcx, typeck, &expr).kind
        // ... of a raw pointer
        && typeck.expr_ty(dereferenced).is_unsafe_ptr()
        {
            let mutbl = Mutability::prefix_str(&mutbl.into());

            let msg = "implicit auto-ref creates a reference to a dereference of a raw pointer";
            cx.struct_span_lint(IMPLICIT_UNSAFE_AUTOREFS, expr.span, msg, |lint| {
                lint
                    .note("creating a reference requires the pointer to be valid and imposes aliasing requirements");

                if let Some(reason) = reason(cx.tcx, implicit_borrow, expr) {
                    lint.note(format!("a reference is implicitly created {reason}"));
                }

                lint
                    .multipart_suggestion(
                        "try using a raw pointer method instead; or if this reference is intentional, make it explicit",
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

/// Peels expressions from `expr` that can map a place.
///
/// For example `(*ptr).field[0]/*<-- built-in index */.field` -> `*ptr`, `f(*ptr)` -> `f(*ptr)`, etc.
fn peel_place_mappers<'tcx>(
    tcx: TyCtxt<'tcx>,
    typeck: &TypeckResults<'tcx>,
    mut expr: &'tcx Expr<'tcx>,
) -> &'tcx Expr<'tcx> {
    loop {
        match expr.kind {
            ExprKind::Index(base, idx)
                if typeck.expr_ty(base).builtin_index() == Some(typeck.expr_ty(expr))
                    && typeck.expr_ty(idx) == tcx.types.usize =>
            {
                expr = &base;
            }
            ExprKind::Field(e, _) => expr = &e,
            _ => break expr,
        }
    }
}

/// Returns `Some(mutability)` if the argument adjustment has implicit borrow in it.
fn has_implicit_borrow(
    Adjustment { kind, .. }: &Adjustment<'_>,
) -> Option<(Mutability, ImplicitBorrow)> {
    match kind {
        &Adjust::Deref(Some(OverloadedDeref { mutbl, .. })) => Some((mutbl, ImplicitBorrow::Deref)),
        &Adjust::Borrow(AutoBorrow::Ref(_, mutbl)) => Some((mutbl.into(), ImplicitBorrow::Borrow)),
        _ => None,
    }
}

enum ImplicitBorrow {
    Deref,
    Borrow,
}

fn reason(tcx: TyCtxt<'_>, borrow_kind: ImplicitBorrow, expr: &Expr<'_>) -> Option<&'static str> {
    match borrow_kind {
        ImplicitBorrow::Deref => Some("because a deref coercion is being applied"),
        ImplicitBorrow::Borrow => {
            let parent = tcx.hir().get(tcx.hir().find_parent_node(expr.hir_id)?);

            let hir::Node::Expr(expr) = parent else {
                return None
            };

            let reason = match expr.kind {
                ExprKind::MethodCall(_, _, _, _) => "to match the method receiver type",
                ExprKind::AssignOp(_, _, _) => {
                    "because a user-defined assignment with an operator is being used"
                }
                ExprKind::Index(_, _) => {
                    "because a user-defined indexing operation is being called"
                }
                _ => return None,
            };

            Some(reason)
        }
    }
}
