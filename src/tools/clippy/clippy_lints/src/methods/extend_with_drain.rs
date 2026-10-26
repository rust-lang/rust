use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::MaybeDef as _;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sym;
use rustc_attr_ir::lang_items::LangItem;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;

use super::EXTEND_WITH_DRAIN;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, extend_receiver: &Expr<'_>, arg: &Expr<'_>) {
    // `expr` is a method call expression which has already been checked to have the method name “extend”.
    // `extend_receiver` is the receiver of that method call, which will be the collection being extended.

    // Check whether the argument of the `extend()` is a `drain()` first, because it’s cheaper than checking types.
    if let ExprKind::MethodCall(src_method, drain_receiver, drain_args, _) = &arg.kind
        && src_method.ident.name == sym::drain
    {
        let extend_collection_ty = cx.typeck_results().expr_ty(extend_receiver).peel_refs();
        let drain_receiver_ty = cx.typeck_results().expr_ty(drain_receiver);

        // Each of these container types has:
        // * A method `fn append(&mut self, other: &mut Self)`.
        // * A method `fn drain(&mut self, /* maybe a range parameter too */)`.
        for (container_type_sym, expect_drain_range_argument) in
            [(sym::Vec, true), (sym::VecDeque, true), (sym::BinaryHeap, false)]
        {
            // Check that the source and destination collections are of the same type.
            if extend_collection_ty.is_diag_item(cx, container_type_sym)
                && drain_receiver_ty.peel_refs().is_diag_item(cx, container_type_sym)
                // Check that the drain range (if there is one) is full, not partial
                && drain_args_are_full_range(cx, drain_args, expect_drain_range_argument)
            {
                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_sugg(
                    cx,
                    EXTEND_WITH_DRAIN,
                    expr.span,
                    format!(
                        "use of `extend` instead of `append` for moving \
                        the full contents of a second `{container_type_sym}`"
                    ),
                    "try",
                    format!(
                        "{}.append({}{})",
                        snippet_with_applicability(cx, extend_receiver.span, "..", &mut applicability),
                        if drain_receiver_ty.is_mutable_ptr() {
                            ""
                        } else {
                            "&mut "
                        },
                        snippet_with_applicability(cx, drain_receiver.span, "..", &mut applicability)
                    ),
                    applicability,
                );
            }
        }
    }
}

/// Check for the correct count of arguments to a `drain()` call, and, if a range is expected,
/// that the range is the full range `..`.
fn drain_args_are_full_range(cx: &LateContext<'_>, args: &[Expr<'_>], expect_drain_range_argument: bool) -> bool {
    match (expect_drain_range_argument, args) {
        (false, []) => true,
        (true, [drain_arg]) => {
            let src_ty_range = cx.typeck_results().expr_ty(drain_arg).peel_refs();
            src_ty_range.is_lang_item(cx, LangItem::RangeFull)
        },
        (_, _) => false,
    }
}
