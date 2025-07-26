use super::EXPLICIT_INTO_ITER_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_trait_method;
use clippy_utils::source::snippet_with_context;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow, AutoBorrowMutability};
use rustc_span::symbol::sym;

#[derive(Clone, Copy)]
enum AdjustKind {
    None,
    Borrow,
    BorrowMut,
    Reborrow,
    ReborrowMut,
}
impl AdjustKind {
    fn borrow(mutbl: AutoBorrowMutability) -> Self {
        match mutbl {
            AutoBorrowMutability::Not => Self::Borrow,
            AutoBorrowMutability::Mut { .. } => Self::BorrowMut,
        }
    }

    fn reborrow(mutbl: AutoBorrowMutability) -> Self {
        match mutbl {
            AutoBorrowMutability::Not => Self::Reborrow,
            AutoBorrowMutability::Mut { .. } => Self::ReborrowMut,
        }
    }

    fn display(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Borrow => "&",
            Self::BorrowMut => "&mut ",
            Self::Reborrow => "&*",
            Self::ReborrowMut => "&mut *",
        }
    }
}

pub(super) fn check(cx: &LateContext<'_>, self_arg: &Expr<'_>, call_expr: &Expr<'_>) {
    if !is_trait_method(cx, call_expr, sym::IntoIterator) {
        return;
    }

    let typeck = cx.typeck_results();
    let self_ty = typeck.expr_ty(self_arg);
    let adjust = match typeck.expr_adjustments(self_arg) {
        [] => AdjustKind::None,
        &[
            Adjustment {
                kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)),
                ..
            },
        ] => AdjustKind::borrow(mutbl),
        &[
            Adjustment {
                kind: Adjust::Deref(_), ..
            },
            Adjustment {
                kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)),
                target,
            },
        ] => {
            if self_ty == target && matches!(mutbl, AutoBorrowMutability::Not) {
                AdjustKind::None
            } else {
                AdjustKind::reborrow(mutbl)
            }
        },
        _ => return,
    };

    let mut applicability = Applicability::MachineApplicable;
    let object = snippet_with_context(cx, self_arg.span, call_expr.span.ctxt(), "_", &mut applicability).0;
    span_lint_and_sugg(
        cx,
        EXPLICIT_INTO_ITER_LOOP,
        call_expr.span,
        "it is more concise to loop over containers instead of using explicit \
            iteration methods",
        "to write this more concisely, try",
        format!("{}{object}", adjust.display()),
        applicability,
    );
}
