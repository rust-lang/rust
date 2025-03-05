use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::get_parent_expr;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_context;
use rustc_ast::ast::LitKind;
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, GenericArg, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::impl_lint_pass;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `size_of::<T>() * 8` when
    /// `T::BITS` is available.
    ///
    /// ### Why is this bad?
    /// Can be written as the shorter `T::BITS`.
    ///
    /// ### Example
    /// ```no_run
    /// size_of::<usize>() * 8;
    /// ```
    /// Use instead:
    /// ```no_run
    /// usize::BITS as usize;
    /// ```
    #[clippy::version = "1.60.0"]
    pub MANUAL_BITS,
    style,
    "manual implementation of `size_of::<T>() * 8` can be simplified with `T::BITS`"
}

pub struct ManualBits {
    msrv: Msrv,
}

impl ManualBits {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(ManualBits => [MANUAL_BITS]);

impl<'tcx> LateLintPass<'tcx> for ManualBits {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Binary(bin_op, left_expr, right_expr) = expr.kind
            && let BinOpKind::Mul = &bin_op.node
            && !expr.span.from_expansion()
            && let ctxt = expr.span.ctxt()
            && left_expr.span.ctxt() == ctxt
            && right_expr.span.ctxt() == ctxt
            && let Some((real_ty_span, resolved_ty, other_expr)) = get_one_size_of_ty(cx, left_expr, right_expr)
            && matches!(resolved_ty.kind(), ty::Int(_) | ty::Uint(_))
            && let ExprKind::Lit(lit) = &other_expr.kind
            && let LitKind::Int(Pu128(8), _) = lit.node
            && self.msrv.meets(cx, msrvs::INTEGER_BITS)
        {
            let mut app = Applicability::MachineApplicable;
            let ty_snip = snippet_with_context(cx, real_ty_span, ctxt, "..", &mut app).0;
            let sugg = create_sugg(cx, expr, format!("{ty_snip}::BITS"));

            span_lint_and_sugg(
                cx,
                MANUAL_BITS,
                expr.span,
                "usage of `size_of::<T>()` to obtain the size of `T` in bits",
                "consider using",
                sugg,
                app,
            );
        }
    }
}

fn get_one_size_of_ty<'tcx>(
    cx: &LateContext<'tcx>,
    expr1: &'tcx Expr<'_>,
    expr2: &'tcx Expr<'_>,
) -> Option<(Span, Ty<'tcx>, &'tcx Expr<'tcx>)> {
    match (get_size_of_ty(cx, expr1), get_size_of_ty(cx, expr2)) {
        (Some((real_ty_span, resolved_ty)), None) => Some((real_ty_span, resolved_ty, expr2)),
        (None, Some((real_ty_span, resolved_ty))) => Some((real_ty_span, resolved_ty, expr1)),
        _ => None,
    }
}

fn get_size_of_ty<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<(Span, Ty<'tcx>)> {
    if let ExprKind::Call(count_func, []) = expr.kind
        && let ExprKind::Path(ref count_func_qpath) = count_func.kind
        && let QPath::Resolved(_, count_func_path) = count_func_qpath
        && let Some(segment_zero) = count_func_path.segments.first()
        && let Some(args) = segment_zero.args
        && let Some(real_ty_span) = args.args.first().map(GenericArg::span)
        && let Some(def_id) = cx.qpath_res(count_func_qpath, count_func.hir_id).opt_def_id()
        && cx.tcx.is_diagnostic_item(sym::mem_size_of, def_id)
    {
        cx.typeck_results()
            .node_args(count_func.hir_id)
            .types()
            .next()
            .map(|resolved_ty| (real_ty_span, resolved_ty))
    } else {
        None
    }
}

fn create_sugg(cx: &LateContext<'_>, expr: &Expr<'_>, base_sugg: String) -> String {
    if let Some(parent_expr) = get_parent_expr(cx, expr) {
        if is_ty_conversion(parent_expr) {
            return base_sugg;
        }

        // These expressions have precedence over casts, the suggestion therefore
        // needs to be wrapped into parentheses
        match parent_expr.kind {
            ExprKind::Unary(..) | ExprKind::AddrOf(..) | ExprKind::MethodCall(..) => {
                return format!("({base_sugg} as usize)");
            },
            _ => {},
        }
    }

    format!("{base_sugg} as usize")
}

fn is_ty_conversion(expr: &Expr<'_>) -> bool {
    if let ExprKind::Cast(..) = expr.kind {
        true
    } else if let ExprKind::MethodCall(path, _, [], _) = expr.kind
        && path.ident.name == sym::try_into
    {
        // This is only called for `usize` which implements `TryInto`. Therefore,
        // we don't have to check here if `self` implements the `TryInto` trait.
        true
    } else {
        false
    }
}
