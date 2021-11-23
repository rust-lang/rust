//! Lint on use of `size_of` or `size_of_val` of T in an expression
//! expecting a count of T

use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{match_def_path, paths};
use if_chain::if_chain;
use rustc_hir::BinOpKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty, TyS, TypeAndMut};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Detects expressions where
    /// `size_of::<T>` or `size_of_val::<T>` is used as a
    /// count of elements of type `T`
    ///
    /// ### Why is this bad?
    /// These functions expect a count
    /// of `T` and not a number of bytes
    ///
    /// ### Example
    /// ```rust,no_run
    /// # use std::ptr::copy_nonoverlapping;
    /// # use std::mem::size_of;
    /// const SIZE: usize = 128;
    /// let x = [2u8; SIZE];
    /// let mut y = [2u8; SIZE];
    /// unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), size_of::<u8>() * SIZE) };
    /// ```
    #[clippy::version = "1.50.0"]
    pub SIZE_OF_IN_ELEMENT_COUNT,
    correctness,
    "using `size_of::<T>` or `size_of_val::<T>` where a count of elements of `T` is expected"
}

declare_lint_pass!(SizeOfInElementCount => [SIZE_OF_IN_ELEMENT_COUNT]);

fn get_size_of_ty(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, inverted: bool) -> Option<Ty<'tcx>> {
    match expr.kind {
        ExprKind::Call(count_func, _func_args) => {
            if_chain! {
                if !inverted;
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
        ExprKind::Binary(op, left, right) if BinOpKind::Mul == op.node => {
            get_size_of_ty(cx, left, inverted).or_else(|| get_size_of_ty(cx, right, inverted))
        },
        ExprKind::Binary(op, left, right) if BinOpKind::Div == op.node => {
            get_size_of_ty(cx, left, inverted).or_else(|| get_size_of_ty(cx, right, !inverted))
        },
        ExprKind::Cast(expr, _) => get_size_of_ty(cx, expr, inverted),
        _ => None,
    }
}

fn get_pointee_ty_and_count_expr(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<(Ty<'tcx>, &'tcx Expr<'tcx>)> {
    const FUNCTIONS: [&[&str]; 8] = [
        &paths::PTR_COPY_NONOVERLAPPING,
        &paths::PTR_COPY,
        &paths::PTR_WRITE_BYTES,
        &paths::PTR_SWAP_NONOVERLAPPING,
        &paths::PTR_SLICE_FROM_RAW_PARTS,
        &paths::PTR_SLICE_FROM_RAW_PARTS_MUT,
        &paths::SLICE_FROM_RAW_PARTS,
        &paths::SLICE_FROM_RAW_PARTS_MUT,
    ];
    const METHODS: [&str; 11] = [
        "write_bytes",
        "copy_to",
        "copy_from",
        "copy_to_nonoverlapping",
        "copy_from_nonoverlapping",
        "add",
        "wrapping_add",
        "sub",
        "wrapping_sub",
        "offset",
        "wrapping_offset",
    ];

    if_chain! {
        // Find calls to ptr::{copy, copy_nonoverlapping}
        // and ptr::{swap_nonoverlapping, write_bytes},
        if let ExprKind::Call(func, [.., count]) = expr.kind;
        if let ExprKind::Path(ref func_qpath) = func.kind;
        if let Some(def_id) = cx.qpath_res(func_qpath, func.hir_id).opt_def_id();
        if FUNCTIONS.iter().any(|func_path| match_def_path(cx, def_id, func_path));

        // Get the pointee type
        if let Some(pointee_ty) = cx.typeck_results().node_substs(func.hir_id).types().next();
        then {
            return Some((pointee_ty, count));
        }
    };
    if_chain! {
        // Find calls to copy_{from,to}{,_nonoverlapping} and write_bytes methods
        if let ExprKind::MethodCall(method_path, _, [ptr_self, .., count], _) = expr.kind;
        let method_ident = method_path.ident.as_str();
        if METHODS.iter().any(|m| *m == &*method_ident);

        // Get the pointee type
        if let ty::RawPtr(TypeAndMut { ty: pointee_ty, .. }) =
            cx.typeck_results().expr_ty(ptr_self).kind();
        then {
            return Some((pointee_ty, count));
        }
    };
    None
}

impl<'tcx> LateLintPass<'tcx> for SizeOfInElementCount {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        const HELP_MSG: &str = "use a count of elements instead of a count of bytes\
            , it already gets multiplied by the size of the type";

        const LINT_MSG: &str = "found a count of bytes \
             instead of a count of elements of `T`";

        if_chain! {
            // Find calls to functions with an element count parameter and get
            // the pointee type and count parameter expression
            if let Some((pointee_ty, count_expr)) = get_pointee_ty_and_count_expr(cx, expr);

            // Find a size_of call in the count parameter expression and
            // check that it's the same type
            if let Some(ty_used_for_size_of) = get_size_of_ty(cx, count_expr, false);
            if TyS::same_type(pointee_ty, ty_used_for_size_of);
            then {
                span_lint_and_help(
                    cx,
                    SIZE_OF_IN_ELEMENT_COUNT,
                    count_expr.span,
                    LINT_MSG,
                    None,
                    HELP_MSG
                );
            }
        };
    }
}
