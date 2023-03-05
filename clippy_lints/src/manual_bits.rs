use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::get_parent_expr;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_context;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, GenericArg, QPath};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for uses of `std::mem::size_of::<T>() * 8` when
    /// `T::BITS` is available.
    ///
    /// ### Why is this bad?
    /// Can be written as the shorter `T::BITS`.
    ///
    /// ### Example
    /// ```rust
    /// std::mem::size_of::<usize>() * 8;
    /// ```
    /// Use instead:
    /// ```rust
    /// usize::BITS as usize;
    /// ```
    #[clippy::version = "1.60.0"]
    pub MANUAL_BITS,
    style,
    "manual implementation of `size_of::<T>() * 8` can be simplified with `T::BITS`"
}

#[derive(Clone)]
pub struct ManualBits {
    msrv: Msrv,
}

impl ManualBits {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(ManualBits => [MANUAL_BITS]);

impl<'tcx> LateLintPass<'tcx> for ManualBits {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !self.msrv.meets(msrvs::MANUAL_BITS) {
            return;
        }

        if_chain! {
            if let ExprKind::Binary(bin_op, left_expr, right_expr) = expr.kind;
            if let BinOpKind::Mul = &bin_op.node;
            if !in_external_macro(cx.sess(), expr.span);
            let ctxt = expr.span.ctxt();
            if left_expr.span.ctxt() == ctxt;
            if right_expr.span.ctxt() == ctxt;
            if let Some((real_ty, resolved_ty, other_expr)) = get_one_size_of_ty(cx, left_expr, right_expr);
            if matches!(resolved_ty.kind(), ty::Int(_) | ty::Uint(_));
            if let ExprKind::Lit(lit) = &other_expr.kind;
            if let LitKind::Int(8, _) = lit.node;
            then {
                let mut app = Applicability::MachineApplicable;
                let ty_snip = snippet_with_context(cx, real_ty.span, ctxt, "..", &mut app).0;
                let sugg = create_sugg(cx, expr, format!("{ty_snip}::BITS"));

                span_lint_and_sugg(
                    cx,
                    MANUAL_BITS,
                    expr.span,
                    "usage of `mem::size_of::<T>()` to obtain the size of `T` in bits",
                    "consider using",
                    sugg,
                    app,
                );
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

fn get_one_size_of_ty<'tcx>(
    cx: &LateContext<'tcx>,
    expr1: &'tcx Expr<'_>,
    expr2: &'tcx Expr<'_>,
) -> Option<(&'tcx rustc_hir::Ty<'tcx>, Ty<'tcx>, &'tcx Expr<'tcx>)> {
    match (get_size_of_ty(cx, expr1), get_size_of_ty(cx, expr2)) {
        (Some((real_ty, resolved_ty)), None) => Some((real_ty, resolved_ty, expr2)),
        (None, Some((real_ty, resolved_ty))) => Some((real_ty, resolved_ty, expr1)),
        _ => None,
    }
}

fn get_size_of_ty<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<(&'tcx rustc_hir::Ty<'tcx>, Ty<'tcx>)> {
    if_chain! {
        if let ExprKind::Call(count_func, _func_args) = expr.kind;
        if let ExprKind::Path(ref count_func_qpath) = count_func.kind;

        if let QPath::Resolved(_, count_func_path) = count_func_qpath;
        if let Some(segment_zero) = count_func_path.segments.get(0);
        if let Some(args) = segment_zero.args;
        if let Some(GenericArg::Type(real_ty)) = args.args.get(0);

        if let Some(def_id) = cx.qpath_res(count_func_qpath, count_func.hir_id).opt_def_id();
        if cx.tcx.is_diagnostic_item(sym::mem_size_of, def_id);
        then {
            cx.typeck_results().node_substs(count_func.hir_id).types().next().map(|resolved_ty| (*real_ty, resolved_ty))
        } else {
            None
        }
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
        && path.ident.name == rustc_span::sym::try_into
    {
        // This is only called for `usize` which implements `TryInto`. Therefore,
        // we don't have to check here if `self` implements the `TryInto` trait.
        true
    } else {
        false
    }
}
