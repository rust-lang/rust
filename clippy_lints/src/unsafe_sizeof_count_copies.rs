//! Lint on unsafe memory copying that use the `size_of` of the pointee type instead of a pointee
//! count

use crate::utils::{match_def_path, paths, span_lint_and_help};
use if_chain::if_chain;
use rustc_hir::BinOpKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty, TyS, TypeAndMut};
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

fn get_size_of_ty(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<Ty<'tcx>> {
    match expr.kind {
        ExprKind::Call(count_func, _func_args) => {
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
            get_size_of_ty(cx, left).or_else(|| get_size_of_ty(cx, right))
        },
        _ => None,
    }
}

fn get_pointee_ty_and_count_expr(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<(Ty<'tcx>, &'tcx Expr<'tcx>)> {
    if_chain! {
        // Find calls to ptr::{copy, copy_nonoverlapping}
        // and ptr::{swap_nonoverlapping, write_bytes},
        if let ExprKind::Call(func, args) = expr.kind;
        if let [_, _, count] = args;
        if let ExprKind::Path(ref func_qpath) = func.kind;
        if let Some(def_id) = cx.qpath_res(func_qpath, func.hir_id).opt_def_id();
        if match_def_path(cx, def_id, &paths::COPY_NONOVERLAPPING)
            || match_def_path(cx, def_id, &paths::COPY)
            || match_def_path(cx, def_id, &paths::WRITE_BYTES)
            || match_def_path(cx, def_id, &paths::PTR_SWAP_NONOVERLAPPING);

        // Get the pointee type
        if let Some(pointee_ty) = cx.typeck_results().node_substs(func.hir_id).types().next();
        then {
            return Some((pointee_ty, count));
        }
    };
    if_chain! {
        // Find calls to copy_{from,to}{,_nonoverlapping} and write_bytes methods
        if let ExprKind::MethodCall(method_path, _, args, _) = expr.kind;
        if let [ptr_self, _, count] = args;
        let method_ident = method_path.ident.as_str();
        if method_ident == "write_bytes" || method_ident == "copy_to" || method_ident == "copy_from"
            || method_ident == "copy_to_nonoverlapping" || method_ident == "copy_from_nonoverlapping";

        // Get the pointee type
        if let ty::RawPtr(TypeAndMut { ty: pointee_ty, mutbl:_mutability }) =
            cx.typeck_results().expr_ty(ptr_self).kind();
        then {
            return Some((pointee_ty, count));
        }
    };
    if_chain! {
        // Find calls to ptr::copy and copy_nonoverlapping
        if let ExprKind::Call(func, args) = expr.kind;
        if let [_data, count] = args;
        if let ExprKind::Path(ref func_qpath) = func.kind;
        if let Some(def_id) = cx.qpath_res(func_qpath, func.hir_id).opt_def_id();
        if match_def_path(cx, def_id, &paths::PTR_SLICE_FROM_RAW_PARTS)
            || match_def_path(cx, def_id, &paths::PTR_SLICE_FROM_RAW_PARTS_MUT);

        // Get the pointee type
        if let Some(pointee_ty) = cx.typeck_results().node_substs(func.hir_id).types().next();
        then {
            return Some((pointee_ty, count));
        }
    };
    None
}

impl<'tcx> LateLintPass<'tcx> for UnsafeSizeofCountCopies {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        const HELP_MSG: &str = "use a count of elements instead of a count of bytes \
            for the count parameter, it already gets multiplied by the size of the pointed to type";

        const LINT_MSG: &str = "unsafe memory copying using a byte count \
            (multiplied by size_of/size_of_val::<T>) instead of a count of T";

        if_chain! {
            // Find calls to unsafe copy functions and get
            // the pointee type and count parameter expression
            if let Some((pointee_ty, count_expr)) = get_pointee_ty_and_count_expr(cx, expr);

            // Find a size_of call in the count parameter expression and
            // check that it's the same type
            if let Some(ty_used_for_size_of) = get_size_of_ty(cx, count_expr);
            if TyS::same_type(pointee_ty, ty_used_for_size_of);
            then {
                span_lint_and_help(
                    cx,
                    UNSAFE_SIZEOF_COUNT_COPIES,
                    expr.span,
                    LINT_MSG,
                    None,
                    HELP_MSG
                );
            }
        };
    }
}
