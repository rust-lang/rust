use crate::{LateContext, LateLintPass, LintContext};

use hir::Expr;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_middle::ty::{self, Ty, TyCtxt, adjustment::{Adjust, AutoBorrow}};

declare_lint! {
    /// The `hidden_reborrow_in_ptr_casts` lint checks for hidden reborrows in `&mut T` -> `*const T` casts.
    ///
    /// ### Example
    ///
    /// ```rust
    /// _ = (&mut 0) as *const _;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    /// 
    /// When casting `&mut T` to `*const T` what actually happens is
    /// 1. `&mut T` is reborrowed into `&T`
    /// 2. `&T` is casted to `*const T`
    /// 
    /// Because this goes through a `&T`, the resulting pointer does not have
    /// write provenance. It is **undefined behaviuor** to write through the
    /// resulting pointer or any pointers derived from it:
    /// 
    /// ```rust
    /// let mut v = 0;
    /// let v_mut = &mut v_mut
    /// let ptr = v_mut as *const i32;
    /// 
    /// // UB
    /// // unsafe { (ptr as *mut i32).write(1) }; 
    /// ```
    /// 
    /// If you don't plan to write through the resulting pointer, you can
    /// suppress this warning adding an explicit cast to a reference:
    /// 
    /// ```rust
    /// let mut v = 0;
    /// let v_mut = &mut v_mut
    /// let ptr = v_mut as &i32 as *const i32;
    /// 
    /// assert_eq!(unsafe { ptr.read() }, 0);
    /// ```
    /// ```rust
    /// let mut v = 0;
    /// let v_mut = &mut v_mut
    /// let ptr = &*v_mut as *const i32;
    /// 
    /// assert_eq!(unsafe { ptr.read() }, 0);
    /// ```
    /// 
    /// If you want to keep the write provenance in a `*const` pointer, cast
    /// to a `*mut` pointer first, to avoid intermidiate shared reference:
    /// 
    /// ```rust
    /// let mut v = 0;
    /// let v_mut = &mut v_mut
    /// let ptr = v_mut as *mut i32 as *const i32;
    /// 
    /// // ok
    /// unsafe { (ptr as *mut i32).write(1) }; 
    /// assert_eq!(v, 1);
    /// ```
    pub HIDDEN_REBORROW_IN_PTR_CASTS,
    Warn,
    "hidden reborrow in a `&mut` -> `*const` cast that strips write provenance"
}

declare_lint_pass!(HiddenReborrowInPtrCasts => [HIDDEN_REBORROW_IN_PTR_CASTS]);

impl<'tcx> LateLintPass<'tcx> for HiddenReborrowInPtrCasts {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let tck = cx.typeck_results();

        let t = tck.expr_ty(expr);

        // t = &mut t_pointee
        if let &ty::Ref(.., t_pointee, hir::Mutability::Mut) = t.kind()
        // // `t -> u` / reference -> pointer adjustments
        && let adjustments = tck.adjustments()
        && let Some(adjustments) = adjustments.get(expr.hir_id) 
        && let [a, b] = &**adjustments
        && let Adjust::Deref(None) = a.kind
        && let Adjust::Borrow(AutoBorrow::RawPtr(_)) = b.kind
        && let u = b.target 
        // u = *const _
        && let ty::RawPtr(ty::TypeAndMut { mutbl: hir::Mutability::Not, ..}) = u.kind()
        // t_pointee is freeze or have variables/generics that make it possibly !Freeze
        && (t_pointee.is_freeze(cx.tcx, cx.param_env) || has_molten_generics(t_pointee, cx.tcx, cx.param_env))
        {
            let msg = "implicit reborrow results in a read-only pointer";
            cx.struct_span_lint(HIDDEN_REBORROW_IN_PTR_CASTS, expr.span, msg, |lint| {
                lint
                    .note("cast of `&mut` reference to `*const` pointer causes an implicit reborrow, which converts the reference to `&`, stripping write provenance")
                    .note("it is UB to write through the resulting pointer, even after casting it to `*mut`")
                    .span_suggestion(
                        expr.span.shrink_to_hi(), 
                        "to save write provenance, cast to `*mut` pointer first", 
                        format!(" as *mut _"),
                        Applicability::MachineApplicable
                    )
                    .span_suggestion(
                        expr.span.shrink_to_hi(), 
                        "to make reborrow explicit, add cast to a shared reference", 
                        format!(" as &_"),
                        Applicability::MachineApplicable
                    )
            })
        }
    }
}

/// Returns `true` if the type is instantiated with at least one generic parameter
/// that itself is a generic parameter (for example of the outer function)
/// and is not freeze.
fn has_molten_generics<'tcx>(ty: Ty<'tcx>, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> bool {
    ty.walk().any(|arg| match arg.unpack() {
        ty::GenericArgKind::Type(t) if matches!(t.kind(), ty::Param(..)) => !t.is_freeze(tcx, param_env),
        _ => false,
    })
}
