use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::Sugg;
use rustc_ast::BorrowKind;
use rustc_errors::{Applicability, Diag};
use rustc_hir::{Expr, ExprKind, Node, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::Adjust;
use rustc_span::sym;

use super::SWAP_WITH_TEMPORARY;

const MSG_TEMPORARY: &str = "this expression returns a temporary value";
const MSG_TEMPORARY_REFMUT: &str = "this is a mutable reference to a temporary value";

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, func: &Expr<'_>, args: &'tcx [Expr<'_>]) {
    if let ExprKind::Path(QPath::Resolved(_, func_path)) = func.kind
        && let Some(func_def_id) = func_path.res.opt_def_id()
        && cx.tcx.is_diagnostic_item(sym::mem_swap, func_def_id)
    {
        match (ArgKind::new(cx, &args[0]), ArgKind::new(cx, &args[1])) {
            (ArgKind::RefMutToTemp(left_temp), ArgKind::RefMutToTemp(right_temp)) => {
                emit_lint_useless(cx, expr, &args[0], &args[1], left_temp, right_temp);
            },
            (ArgKind::RefMutToTemp(left_temp), right) => emit_lint_assign(cx, expr, &right, &args[0], left_temp),
            (left, ArgKind::RefMutToTemp(right_temp)) => emit_lint_assign(cx, expr, &left, &args[1], right_temp),
            _ => {},
        }
    }
}

enum ArgKind<'tcx> {
    // Mutable reference to a place, coming from a macro, and number of dereferences to use
    RefMutToPlaceAsMacro(&'tcx Expr<'tcx>, usize),
    // Place behind a mutable reference, and number of dereferences to use
    RefMutToPlace(&'tcx Expr<'tcx>, usize),
    // Temporary value behind a mutable reference
    RefMutToTemp(&'tcx Expr<'tcx>),
    // Any other case
    Expr(&'tcx Expr<'tcx>),
}

impl<'tcx> ArgKind<'tcx> {
    /// Build a new `ArgKind` from `arg`. There must be no false positive when returning a
    /// `ArgKind::RefMutToTemp` variant, as this may cause a spurious lint to be emitted.
    fn new(cx: &LateContext<'tcx>, arg: &'tcx Expr<'tcx>) -> Self {
        if let ExprKind::AddrOf(BorrowKind::Ref, _, target) = arg.kind
            && let adjustments = cx.typeck_results().expr_adjustments(arg)
            && adjustments
                .first()
                .is_some_and(|adj| matches!(adj.kind, Adjust::Deref(None)))
            && adjustments
                .last()
                .is_some_and(|adj| matches!(adj.kind, Adjust::Borrow(_)))
        {
            let extra_derefs = adjustments[1..adjustments.len() - 1]
                .iter()
                .filter(|adj| matches!(adj.kind, Adjust::Deref(_)))
                .count();
            // If a deref is used, `arg` might be a place expression. For example, a mutex guard
            // would dereference into the mutex content which is probably not temporary.
            if target.is_syntactic_place_expr() || extra_derefs > 0 {
                if arg.span.from_expansion() {
                    ArgKind::RefMutToPlaceAsMacro(arg, extra_derefs)
                } else {
                    ArgKind::RefMutToPlace(target, extra_derefs)
                }
            } else {
                ArgKind::RefMutToTemp(target)
            }
        } else {
            ArgKind::Expr(arg)
        }
    }
}

// Emits a note either on the temporary expression if it can be found in the same context as the
// base and returns `true`, or on the mutable reference to the temporary expression otherwise and
// returns `false`.
fn emit_note(diag: &mut Diag<'_, ()>, base: &Expr<'_>, expr: &Expr<'_>, expr_temp: &Expr<'_>) -> bool {
    if base.span.eq_ctxt(expr.span) {
        diag.span_note(expr_temp.span.source_callsite(), MSG_TEMPORARY);
        true
    } else {
        diag.span_note(expr.span.source_callsite(), MSG_TEMPORARY_REFMUT);
        false
    }
}

fn emit_lint_useless(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    left: &Expr<'_>,
    right: &Expr<'_>,
    left_temp: &Expr<'_>,
    right_temp: &Expr<'_>,
) {
    span_lint_and_then(
        cx,
        SWAP_WITH_TEMPORARY,
        expr.span,
        "swapping temporary values has no effect",
        |diag| {
            emit_note(diag, expr, left, left_temp);
            emit_note(diag, expr, right, right_temp);
        },
    );
}

fn emit_lint_assign(cx: &LateContext<'_>, expr: &Expr<'_>, target: &ArgKind<'_>, reftemp: &Expr<'_>, temp: &Expr<'_>) {
    span_lint_and_then(
        cx,
        SWAP_WITH_TEMPORARY,
        expr.span,
        "swapping with a temporary value is inefficient",
        |diag| {
            if !emit_note(diag, expr, reftemp, temp) {
                return;
            }

            // Make the suggestion only when the original `swap()` call is a statement
            // or the last expression in a block.
            if matches!(cx.tcx.parent_hir_node(expr.hir_id), Node::Stmt(..) | Node::Block(..)) {
                let mut applicability = Applicability::MachineApplicable;
                let ctxt = expr.span.ctxt();
                let assign_target = match target {
                    ArgKind::Expr(target) => Sugg::hir_with_context(cx, target, ctxt, "_", &mut applicability).deref(),
                    ArgKind::RefMutToPlaceAsMacro(arg, derefs) => (0..*derefs).fold(
                        Sugg::hir_with_context(cx, arg, ctxt, "_", &mut applicability).deref(),
                        |sugg, _| sugg.deref(),
                    ),
                    ArgKind::RefMutToPlace(target, derefs) => (0..*derefs).fold(
                        Sugg::hir_with_context(cx, target, ctxt, "_", &mut applicability),
                        |sugg, _| sugg.deref(),
                    ),
                    ArgKind::RefMutToTemp(_) => unreachable!(),
                };
                let assign_source = Sugg::hir_with_context(cx, temp, ctxt, "_", &mut applicability);
                diag.span_suggestion(
                    expr.span,
                    "use assignment instead",
                    format!("{assign_target} = {assign_source}"),
                    applicability,
                );
            }
        },
    );
}
