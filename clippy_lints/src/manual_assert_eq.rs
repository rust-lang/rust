use clippy_utils::consts::ConstEvalCtxt;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::{PanicCall, find_assert_args, root_macro_call_first_node};
use clippy_utils::source::walk_span_to_context;
use clippy_utils::ty::{deref_chain, implements_trait};
use clippy_utils::{is_in_const_context, sym};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext as _, declare_lint_pass};
use rustc_middle::ty::{self, Ty};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `assert!` and `debug_assert!` that consist of only an (in)equality check
    ///
    /// ### Why is this bad?
    /// `assert_{eq,ne}!` and `debug_assert_{eq,ne}!` achieves the same goal, and provides some
    /// additional debug information
    ///
    /// ### Known problems
    /// This lint cannot determine how large the `Debug` output of the compared values will be.
    /// To avoid producing excessively large assertion output, it ignores comparisons involving
    /// byte-slice-like types. These include byte slices and types that dereference to a byte slice
    /// or implement `AsRef<[u8]>` without also implementing `AsRef<str>`.
    ///
    /// ### Example
    /// ```no_run
    /// assert!(2 * 2 == 4);
    /// assert!(2 * 2 != 5);
    /// debug_assert!(2 * 2 == 4);
    /// debug_assert!(2 * 2 != 5);
    /// ```
    /// Use instead:
    /// ```no_run
    /// assert_eq!(2 * 2, 4);
    /// assert_ne!(2 * 2, 5);
    /// debug_assert_eq!(2 * 2, 4);
    /// debug_assert_ne!(2 * 2, 5);
    /// ```
    #[clippy::version = "1.97.0"]
    pub MANUAL_ASSERT_EQ,
    pedantic,
    "checks for assertions consisting of an (in)equality check"
}

declare_lint_pass!(ManualAssertEq => [MANUAL_ASSERT_EQ]);

#[derive(Clone, Copy, PartialEq, Eq)]
enum EqKind {
    Eq,
    Ne,
}

impl EqKind {
    fn postfix(self) -> &'static str {
        match self {
            Self::Eq => "_eq",
            Self::Ne => "_ne",
        }
    }
}

impl LateLintPass<'_> for ManualAssertEq {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let Some(macro_call) = root_macro_call_first_node(cx, expr)
            && let macro_name = match cx.tcx.get_diagnostic_name(macro_call.def_id) {
                Some(sym::assert_macro) => "assert",
                Some(sym::debug_assert_macro) => "debug_assert",
                _ => return,
            }
            && !is_in_const_context(cx)
            && let Some((cond, panic_expn)) = find_assert_args(cx, expr, macro_call.expn)
            // Don't lint if the user has a painstakingly written assertion message
            && !matches!(panic_expn, PanicCall::Display(_) | PanicCall::Format(_))
            && let ExprKind::Binary(op, lhs, rhs) = cond.kind
            && let eq_kind = match op.node {
                BinOpKind::Eq => EqKind::Eq,
                BinOpKind::Ne => EqKind::Ne,
                _ => return,
            }
            && !cond.span.from_expansion()
            && let Some(debug_trait) = cx.tcx.get_diagnostic_item(sym::Debug)
            && let lhs_ty = cx.typeck_results().expr_ty(lhs)
            && let rhs_ty = cx.typeck_results().expr_ty(rhs)
            // Can't print the values unless the types implement `Debug`
            && implements_trait(cx, lhs_ty, debug_trait, &[])
            && implements_trait(cx, rhs_ty, debug_trait, &[])
            // Printing raw pointers isn't very useful
            && !lhs_ty.is_raw_ptr()
            && !rhs_ty.is_raw_ptr()
            // Byte buffers can be large and their debug output is rarely useful
            && !is_byte_slice_like(cx, lhs_ty)
            && !is_byte_slice_like(cx, rhs_ty)
            // The output of `(debug_)assert_eq` isn't very useful when one of the sides is a constant value
            && if eq_kind == EqKind::Ne {
                   let ecx = ConstEvalCtxt::new(cx);
                   ecx.eval(lhs).is_none() && ecx.eval(rhs).is_none()
               } else {
                   true
            }
        {
            span_lint_and_then(
                cx,
                MANUAL_ASSERT_EQ,
                macro_call.span,
                format!("used `{macro_name}!` with an equality comparison"),
                |diag| {
                    let postfix = eq_kind.postfix();
                    let new_name = format_args!("{macro_name}{postfix}");
                    let msg = format!("replace it with `{new_name}!(..)`");

                    let ctxt = cond.span.ctxt();
                    if let Some(lhs_span) = walk_span_to_context(lhs.span, ctxt)
                        && let Some(rhs_span) = walk_span_to_context(rhs.span, ctxt)
                    {
                        let macro_name_span = cx.sess().source_map().span_until_char(macro_call.span, '!');
                        let eq_span = cond.span.with_lo(lhs_span.hi()).with_hi(rhs_span.lo());
                        let suggestions = vec![
                            (macro_name_span.shrink_to_hi(), postfix.to_string()),
                            (eq_span, ", ".to_string()),
                        ];

                        diag.multipart_suggestion(msg, suggestions, Applicability::MachineApplicable);
                    } else {
                        diag.span_help(expr.span, msg);
                    }
                },
            );
        }
    }
}

fn is_byte_slice_like<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
    let byte_slice = Ty::new_slice(cx.tcx, cx.tcx.types.u8);
    let ty = ty.peel_refs();

    if ty == byte_slice {
        return true;
    }
    if matches!(ty.kind(), ty::Adt(..))
        && cx.tcx.get_diagnostic_item(sym::AsRef).is_some_and(|trait_id| {
            implements_trait(cx, ty, trait_id, &[byte_slice.into()])
                && !implements_trait(cx, ty, trait_id, &[cx.tcx.types.str_.into()])
        })
    {
        return true;
    }

    for (depth, ty) in deref_chain(cx, ty).enumerate().skip(1) {
        if !cx.tcx.recursion_limit().value_within_limit(depth) {
            return false;
        }
        if ty == byte_slice {
            return true;
        }
    }

    false
}
