//! Lint on unsafe memory copying that use the `size_of` of the pointee type instead of a pointee
//! count

use crate::utils::{match_def_path, paths, span_lint_and_help};
use if_chain::if_chain;
use rustc_hir::BinOpKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{Ty as TyM, TyS};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Detects expressions where
    /// size_of::<T> is used as the count argument to unsafe
    /// memory copying functions like ptr::copy and
    /// ptr::copy_nonoverlapping where T is the pointee type
    /// of the pointers used
    ///
    /// **Why is this bad?** These functions expect a count
    /// of T and not a number of bytes, which can lead to
    /// copying the incorrect amount of bytes, which can
    /// result in Undefined Behaviour
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,no_run
    /// # use std::ptr::copy_nonoverlapping;
    /// # use std::mem::size_of;
    ///
    /// const SIZE: usize = 128;
    /// let x = [2u8; SIZE];
    /// let mut y = [2u8; SIZE];
    /// unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), size_of::<u8>() * SIZE) };
    /// ```
    pub UNSAFE_SIZEOF_COUNT_COPIES,
    correctness,
    "unsafe memory copying using a byte count instead of a count of T"
}

declare_lint_pass!(UnsafeSizeofCountCopies => [UNSAFE_SIZEOF_COUNT_COPIES]);

fn get_size_of_ty(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<TyM<'tcx>> {
    match &expr.kind {
        ExprKind::Call(ref count_func, _func_args) => {
            if_chain! {
                if let ExprKind::Path(ref count_func_qpath) = count_func.kind;
                if let Some(def_id) = cx.qpath_res(count_func_qpath, count_func.hir_id).opt_def_id();
                if match_def_path(cx, def_id, &paths::MEM_SIZE_OF)
                    || match_def_path(cx, def_id, &paths::MEM_SIZE_OF_VAL);
                then {
                    cx.typeck_results().node_substs(count_func.hir_id).types().next()
                } else {
                    None
                }
            }
        },
        ExprKind::Binary(op, left, right) if BinOpKind::Mul == op.node || BinOpKind::Div == op.node => {
            get_size_of_ty(cx, &*left).or_else(|| get_size_of_ty(cx, &*right))
        },
        _ => None,
    }
}

impl<'tcx> LateLintPass<'tcx> for UnsafeSizeofCountCopies {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            // Find calls to ptr::copy and copy_nonoverlapping
            if let ExprKind::Call(ref func, ref func_args) = expr.kind;
            if let ExprKind::Path(ref func_qpath) = func.kind;
            if let Some(def_id) = cx.qpath_res(func_qpath, func.hir_id).opt_def_id();
            if match_def_path(cx, def_id, &paths::COPY_NONOVERLAPPING)
                || match_def_path(cx, def_id, &paths::COPY);

            // Get the pointee type
            let _substs = cx.typeck_results().node_substs(func.hir_id);
            if let Some(pointee_ty) = cx.typeck_results().node_substs(func.hir_id).types().next();

            // Find a size_of call in the count parameter expression and
            // check that it's the same type
            if let [_src, _dest, count] = &**func_args;
            if let Some(ty_used_for_size_of) = get_size_of_ty(cx, count);
            if TyS::same_type(pointee_ty, ty_used_for_size_of);
            then {
                span_lint_and_help(
                    cx,
                    UNSAFE_SIZEOF_COUNT_COPIES,
                    expr.span,
                    "unsafe memory copying using a byte count (Multiplied by size_of::<T>) \
                    instead of a count of T",
                    None,
                    "use a count of elements instead of a count of bytes for the count parameter, \
                    it already gets multiplied by the size of the pointed to type"
                );
            }
        };
    }
}
