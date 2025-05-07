use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Mutability, QPath, TyKind};
use rustc_hir_pretty::qpath_to_string;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::sym;

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

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, msrv: Msrv) {
    if let ExprKind::Cast(cast_expr, cast_to_hir_ty) = expr.kind
        && let (cast_from, cast_to) = (cx.typeck_results().expr_ty(cast_expr), cx.typeck_results().expr_ty(expr))
        && let ty::RawPtr(_, from_mutbl) = cast_from.kind()
        && let ty::RawPtr(to_pointee_ty, to_mutbl) = cast_to.kind()
        && matches!((from_mutbl, to_mutbl),
            (Mutability::Not, Mutability::Not) | (Mutability::Mut, Mutability::Mut))
        // The `U` in `pointer::cast` have to be `Sized`
        // as explained here: https://github.com/rust-lang/rust/issues/60602.
        && to_pointee_ty.is_sized(cx.tcx, cx.typing_env())
        && msrv.meets(cx, msrvs::POINTER_CAST)
    {
        let mut app = Applicability::MachineApplicable;
        let turbofish = match &cast_to_hir_ty.kind {
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
        let omit_cast = if let ExprKind::Call(func, []) = cast_expr.kind
            && let ExprKind::Path(ref qpath @ QPath::Resolved(None, path)) = func.kind
            && let Some(method_defid) = path.res.opt_def_id()
        {
            if cx.tcx.is_diagnostic_item(sym::ptr_null, method_defid) {
                OmitFollowedCastReason::Null(qpath)
            } else if cx.tcx.is_diagnostic_item(sym::ptr_null_mut, method_defid) {
                OmitFollowedCastReason::NullMut(qpath)
            } else {
                OmitFollowedCastReason::None
            }
        } else {
            OmitFollowedCastReason::None
        };

        let (help, final_suggestion) = if let Some(method) = omit_cast.corresponding_item() {
            // don't force absolute path
            let method = qpath_to_string(&cx.tcx, method);
            ("try call directly", format!("{method}{turbofish}()"))
        } else {
            let cast_expr_sugg = Sugg::hir_with_applicability(cx, cast_expr, "_", &mut app);

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
