use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::declare_lint_pass;
use rustc_span::sym;

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

fn get_size_of_ty<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, inverted: bool) -> Option<Ty<'tcx>> {
    match expr.kind {
        ExprKind::Call(count_func, _) => {
            if !inverted
                && let ExprKind::Path(ref count_func_qpath) = count_func.kind
                && let Some(def_id) = cx.qpath_res(count_func_qpath, count_func.hir_id).opt_def_id()
                && matches!(
                    cx.tcx.get_diagnostic_name(def_id),
                    Some(sym::mem_size_of | sym::mem_size_of_val)
                )
            {
                cx.typeck_results().node_args(count_func.hir_id).types().next()
            } else {
                None
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

fn get_pointee_ty_and_count_expr<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
) -> Option<(Ty<'tcx>, &'tcx Expr<'tcx>)> {
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

    if let ExprKind::Call(func, [.., count]) = expr.kind
        // Find calls to ptr::{copy, copy_nonoverlapping}
        // and ptr::{swap_nonoverlapping, write_bytes},
        && let ExprKind::Path(ref func_qpath) = func.kind
        && let Some(def_id) = cx.qpath_res(func_qpath, func.hir_id).opt_def_id()
        && matches!(cx.tcx.get_diagnostic_name(def_id), Some(
            sym::ptr_copy
            | sym::ptr_copy_nonoverlapping
            | sym::ptr_slice_from_raw_parts
            | sym::ptr_slice_from_raw_parts_mut
            | sym::ptr_swap_nonoverlapping
            | sym::ptr_write_bytes
            | sym::slice_from_raw_parts
            | sym::slice_from_raw_parts_mut
        ))

        // Get the pointee type
        && let Some(pointee_ty) = cx.typeck_results().node_args(func.hir_id).types().next()
    {
        return Some((pointee_ty, count));
    };
    if let ExprKind::MethodCall(method_path, ptr_self, [.., count], _) = expr.kind
        // Find calls to copy_{from,to}{,_nonoverlapping} and write_bytes methods
        && let method_ident = method_path.ident.as_str()
        && METHODS.iter().any(|m| *m == method_ident)

        // Get the pointee type
        && let ty::RawPtr(pointee_ty, _) =
            cx.typeck_results().expr_ty(ptr_self).kind()
    {
        return Some((*pointee_ty, count));
    };
    None
}

impl<'tcx> LateLintPass<'tcx> for SizeOfInElementCount {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        const HELP_MSG: &str = "use a count of elements instead of a count of bytes\
            , it already gets multiplied by the size of the type";

        const LINT_MSG: &str = "found a count of bytes \
             instead of a count of elements of `T`";

        if let Some((pointee_ty, count_expr)) = get_pointee_ty_and_count_expr(cx, expr)
            // Find calls to functions with an element count parameter and get
            // the pointee type and count parameter expression

            // Find a size_of call in the count parameter expression and
            // check that it's the same type
            && let Some(ty_used_for_size_of) = get_size_of_ty(cx, count_expr, false)
            && pointee_ty == ty_used_for_size_of
        {
            span_lint_and_help(cx, SIZE_OF_IN_ELEMENT_COUNT, count_expr.span, LINT_MSG, None, HELP_MSG);
        };
    }
}
