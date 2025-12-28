use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::res::MaybeDef;
use clippy_utils::{eq_expr_value, sym};
use rustc_hir::{Expr, ExprKind, LangItem, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::sym as rustc_sym;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for usages of `Vec::from_raw_parts` and `String::from_raw_parts`
    /// where the same expression is used for the length and the capacity.
    ///
    /// ### Why is this bad?
    ///
    /// If the same expression is being passed for the length and
    /// capacity, it is most likely a semantic error. In the case of a
    /// Vec, for example, the only way to end up with one that has
    /// the same length and capacity is by going through a boxed slice,
    /// e.g. `Box::from(some_vec)`, which shrinks the capacity to match
    /// the length.
    ///
    /// ### Example
    ///
    /// ```no_run
    /// #![feature(vec_into_raw_parts)]
    /// let mut original: Vec::<i32> = Vec::with_capacity(20);
    /// original.extend([1, 2, 3, 4, 5]);
    ///
    /// let (ptr, mut len, cap) = original.into_raw_parts();
    ///
    /// // I will add three more integers:
    /// unsafe {
    ///    let ptr = ptr as *mut i32;
    ///
    ///    for i in 6..9 {
    ///        *ptr.add(i - 1) = i as i32;
    ///        len += 1;
    ///    }
    /// }
    ///
    /// // But I forgot the capacity was separate from the length:
    /// let reconstructed = unsafe { Vec::from_raw_parts(ptr, len, len) };
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #![feature(vec_into_raw_parts)]
    /// let mut original: Vec::<i32> = Vec::with_capacity(20);
    /// original.extend([1, 2, 3, 4, 5]);
    ///
    /// let (ptr, mut len, cap) = original.into_raw_parts();
    ///
    /// // I will add three more integers:
    /// unsafe {
    ///    let ptr = ptr as *mut i32;
    ///
    ///    for i in 6..9 {
    ///        *ptr.add(i - 1) = i as i32;
    ///        len += 1;
    ///    }
    /// }
    ///
    /// // This time, leverage the previously saved capacity:
    /// let reconstructed = unsafe { Vec::from_raw_parts(ptr, len, cap) };
    /// ```
    #[clippy::version = "1.93.0"]
    pub SAME_LENGTH_AND_CAPACITY,
    pedantic,
    "`from_raw_parts` with same length and capacity"
}
declare_lint_pass!(SameLengthAndCapacity => [SAME_LENGTH_AND_CAPACITY]);

impl<'tcx> LateLintPass<'tcx> for SameLengthAndCapacity {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Call(path_expr, args) = expr.kind
            && let ExprKind::Path(QPath::TypeRelative(ty, fn_path)) = path_expr.kind
            && fn_path.ident.name == sym::from_raw_parts
            && args.len() >= 3
            && eq_expr_value(cx, &args[1], &args[2])
        {
            let middle_ty = cx.typeck_results().node_type(ty.hir_id);
            if middle_ty.is_diag_item(cx, rustc_sym::Vec) {
                span_lint_and_help(
                    cx,
                    SAME_LENGTH_AND_CAPACITY,
                    expr.span,
                    "usage of `Vec::from_raw_parts` with the same expression for length and capacity",
                    None,
                    "try `Box::from(slice::from_raw_parts(...)).into::<Vec<_>>()`",
                );
            } else if middle_ty.is_lang_item(cx, LangItem::String) {
                span_lint_and_help(
                    cx,
                    SAME_LENGTH_AND_CAPACITY,
                    expr.span,
                    "usage of `String::from_raw_parts` with the same expression for length and capacity",
                    None,
                    "try `String::from(str::from_utf8_unchecked(slice::from_raw_parts(...)))`",
                );
            }
        }
    }
}
