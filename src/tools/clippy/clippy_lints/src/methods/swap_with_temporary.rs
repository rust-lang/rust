use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::Sugg;
use rustc_ast::BorrowKind;
use rustc_errors::{Applicability, Diag};
use rustc_hir::{Expr, ExprKind, Node, QPath};
use rustc_lint::LateContext;
use rustc_span::sym;

use super::SWAP_WITH_TEMPORARY;

const MSG_TEMPORARY: &str = "this expression returns a temporary value";
const MSG_TEMPORARY_REFMUT: &str = "this is a mutable reference to a temporary value";

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, func: &Expr<'_>, args: &[Expr<'_>]) {
    if let ExprKind::Path(QPath::Resolved(_, func_path)) = func.kind
        && let Some(func_def_id) = func_path.res.opt_def_id()
        && cx.tcx.is_diagnostic_item(sym::mem_swap, func_def_id)
    {
        match (ArgKind::new(&args[0]), ArgKind::new(&args[1])) {
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
    // Mutable reference to a place, coming from a macro
    RefMutToPlaceAsMacro(&'tcx Expr<'tcx>),
    // Place behind a mutable reference
    RefMutToPlace(&'tcx Expr<'tcx>),
    // Temporary value behind a mutable reference
    RefMutToTemp(&'tcx Expr<'tcx>),
    // Any other case
    Expr(&'tcx Expr<'tcx>),
}

impl<'tcx> ArgKind<'tcx> {
    fn new(arg: &'tcx Expr<'tcx>) -> Self {
        if let ExprKind::AddrOf(BorrowKind::Ref, _, target) = arg.kind {
            if target.is_syntactic_place_expr() {
                if arg.span.from_expansion() {
                    ArgKind::RefMutToPlaceAsMacro(arg)
                } else {
                    ArgKind::RefMutToPlace(target)
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
                    ArgKind::Expr(target) | ArgKind::RefMutToPlaceAsMacro(target) => {
                        Sugg::hir_with_context(cx, target, ctxt, "_", &mut applicability).deref()
                    },
                    ArgKind::RefMutToPlace(target) => Sugg::hir_with_context(cx, target, ctxt, "_", &mut applicability),
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
