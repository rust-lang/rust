use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_from_proc_macro;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, Expr, ExprKind, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::{Span, sym};

use super::PTR_AS_PTR;

enum OmitFollowedCastReason<'a> {
    None,
    Null(&'a QPath<'a>),
    NullMut(&'a QPath<'a>),
}

impl OmitFollowedCastReason<'_> {
    fn corresponding_item(&self) -> Option<&QPath<'_>> {
        match self {
            OmitFollowedCastReason::None => None,
            OmitFollowedCastReason::Null(x) | OmitFollowedCastReason::NullMut(x) => Some(*x),
        }
    }
}

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    cast_from_expr: &Expr<'_>,
    cast_from: Ty<'_>,
    cast_to_hir: &hir::Ty<'_>,
    cast_to: Ty<'tcx>,
    msrv: Msrv,
) {
    if let ty::RawPtr(_, from_mutbl) = cast_from.kind()
        && let ty::RawPtr(to_pointee_ty, to_mutbl) = cast_to.kind()
        && from_mutbl == to_mutbl
        // The `U` in `pointer::cast` have to be `Sized`
        // as explained here: https://github.com/rust-lang/rust/issues/60602.
        && to_pointee_ty.is_sized(cx.tcx, cx.typing_env())
        && msrv.meets(cx, msrvs::POINTER_CAST)
        && !is_from_proc_macro(cx, expr)
    {
        let mut app = Applicability::MachineApplicable;
        let turbofish = match &cast_to_hir.kind {
            TyKind::Infer(()) => String::new(),
            TyKind::Ptr(mut_ty) => {
                if matches!(mut_ty.ty.kind, TyKind::Infer(())) {
                    String::new()
                } else {
                    format!(
                        "::<{}>",
                        snippet_with_applicability(cx, mut_ty.ty.span, "/* type */", &mut app)
                    )
                }
            },
            _ => return,
        };

        // following `cast` does not compile because it fails to infer what type is expected
        // as type argument to `std::ptr::ptr_null` or `std::ptr::ptr_null_mut`, so
        // we omit following `cast`:
        let omit_cast = if let ExprKind::Call(func, []) = cast_from_expr.kind
            && let ExprKind::Path(ref qpath @ QPath::Resolved(None, path)) = func.kind
            && let Some(method_defid) = path.res.opt_def_id()
        {
            match cx.tcx.get_diagnostic_name(method_defid) {
                Some(sym::ptr_null) => OmitFollowedCastReason::Null(qpath),
                Some(sym::ptr_null_mut) => OmitFollowedCastReason::NullMut(qpath),
                _ => OmitFollowedCastReason::None,
            }
        } else {
            OmitFollowedCastReason::None
        };

        let (help, final_suggestion) = if let Some(method) = omit_cast.corresponding_item() {
            // don't force absolute path
            let method = snippet_with_applicability(cx, qpath_span_without_turbofish(method), "..", &mut app);
            ("try call directly", format!("{method}{turbofish}()"))
        } else {
            let cast_expr_sugg = Sugg::hir_with_context(cx, cast_from_expr, expr.span.ctxt(), "_", &mut app);

            (
                "try `pointer::cast`, a safer alternative",
                format!("{}.cast{turbofish}()", cast_expr_sugg.maybe_paren()),
            )
        };

        span_lint_and_sugg(
            cx,
            PTR_AS_PTR,
            expr.span,
            "`as` casting between raw pointers without changing their constness",
            help,
            final_suggestion,
            app,
        );
    }
}

fn qpath_span_without_turbofish(qpath: &QPath<'_>) -> Span {
    if let QPath::Resolved(_, path) = qpath
        && let [.., last_ident] = path.segments
        && last_ident.args.is_some()
    {
        return qpath.span().shrink_to_lo().to(last_ident.ident.span);
    }

    qpath.span()
}
