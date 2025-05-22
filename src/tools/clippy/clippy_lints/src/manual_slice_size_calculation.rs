use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_context;
use clippy_utils::{expr_or_init, is_in_const_context, std_or_core};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// When `a` is `&[T]`, detect `a.len() * size_of::<T>()` and suggest `size_of_val(a)`
    /// instead.
    ///
    /// ### Why is this better?
    /// * Shorter to write
    /// * Removes the need for the human and the compiler to worry about overflow in the
    ///   multiplication
    /// * Potentially faster at runtime as rust emits special no-wrapping flags when it
    ///   calculates the byte length
    /// * Less turbofishing
    ///
    /// ### Example
    /// ```no_run
    /// # let data : &[i32] = &[1, 2, 3];
    /// let newlen = data.len() * size_of::<i32>();
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let data : &[i32] = &[1, 2, 3];
    /// let newlen = size_of_val(data);
    /// ```
    #[clippy::version = "1.70.0"]
    pub MANUAL_SLICE_SIZE_CALCULATION,
    complexity,
    "manual slice size calculation"
}
impl_lint_pass!(ManualSliceSizeCalculation => [MANUAL_SLICE_SIZE_CALCULATION]);

pub struct ManualSliceSizeCalculation {
    msrv: Msrv,
}

impl ManualSliceSizeCalculation {
    pub fn new(conf: &Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for ManualSliceSizeCalculation {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Binary(ref op, left, right) = expr.kind
            && BinOpKind::Mul == op.node
            && !expr.span.from_expansion()
            && let Some((receiver, refs_count)) = simplify(cx, left, right)
            && (!is_in_const_context(cx) || self.msrv.meets(cx, msrvs::CONST_SIZE_OF_VAL))
        {
            let ctxt = expr.span.ctxt();
            let mut app = Applicability::MachineApplicable;
            let deref = if refs_count > 0 {
                "*".repeat(refs_count - 1)
            } else {
                "&".into()
            };
            let val_name = snippet_with_context(cx, receiver.span, ctxt, "slice", &mut app).0;
            let Some(sugg) = std_or_core(cx) else { return };

            span_lint_and_sugg(
                cx,
                MANUAL_SLICE_SIZE_CALCULATION,
                expr.span,
                "manual slice size calculation",
                "try",
                format!("{sugg}::mem::size_of_val({deref}{val_name})"),
                app,
            );
        }
    }
}

fn simplify<'tcx>(
    cx: &LateContext<'tcx>,
    expr1: &'tcx Expr<'tcx>,
    expr2: &'tcx Expr<'tcx>,
) -> Option<(&'tcx Expr<'tcx>, usize)> {
    let expr1 = expr_or_init(cx, expr1);
    let expr2 = expr_or_init(cx, expr2);

    simplify_half(cx, expr1, expr2).or_else(|| simplify_half(cx, expr2, expr1))
}

fn simplify_half<'tcx>(
    cx: &LateContext<'tcx>,
    expr1: &'tcx Expr<'tcx>,
    expr2: &'tcx Expr<'tcx>,
) -> Option<(&'tcx Expr<'tcx>, usize)> {
    if !expr1.span.from_expansion()
        // expr1 is `[T1].len()`?
        && let ExprKind::MethodCall(method_path, receiver, [], _) = expr1.kind
        && method_path.ident.name == sym::len
        && let receiver_ty = cx.typeck_results().expr_ty(receiver)
        && let (receiver_ty, refs_count) = clippy_utils::ty::walk_ptrs_ty_depth(receiver_ty)
        && let ty::Slice(ty1) = receiver_ty.kind()
        // expr2 is `size_of::<T2>()`?
        && let ExprKind::Call(func, []) = expr2.kind
        && let ExprKind::Path(ref func_qpath) = func.kind
        && let Some(def_id) = cx.qpath_res(func_qpath, func.hir_id).opt_def_id()
        && cx.tcx.is_diagnostic_item(sym::mem_size_of, def_id)
        && let Some(ty2) = cx.typeck_results().node_args(func.hir_id).types().next()
        // T1 == T2?
        && *ty1 == ty2
    {
        Some((receiver, refs_count))
    } else {
        None
    }
}
