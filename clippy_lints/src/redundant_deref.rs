// use crate::utils::{get_parent_expr, snippet_with_applicability, span_lint_and_sugg};
// use if_chain::if_chain;
// use rustc_errors::Applicability;
// use rustc_hir::{Expr, ExprKind, UnOp};
// use rustc_lint::{LateContext, LateLintPass, LintContext};
// use rustc_middle::lint::in_external_macro;
// use rustc_session::{declare_lint_pass, declare_tool_lint};

// declare_clippy_lint! {
//     /// **What it does:** Checks for uses of the dereference operator which would be covered by
//     /// auto-dereferencing.
//     ///
//     /// **Why is this bad?** This unnecessarily complicates the code.
//     ///
//     /// **Known problems:** None.
//     ///
//     /// **Example:**
//     ///
//     /// ```rust
//     /// fn foo(_: &str) {}
//     /// foo(&*String::new())
//     /// ```
//     /// Use instead:
//     /// ```rust
//     /// fn foo(_: &str) {}
//     /// foo(&String::new())
//     /// ```
//     pub REDUNDANT_DEREF,
//     style,
//     "default lint description"
// }

// declare_lint_pass!(RedundantDeref => [REDUNDANT_DEREF]);

// impl LateLintPass<'_> for RedundantDeref {
//     fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
//         if_chain! {
//             if let ExprKind::AddrOf(_, _, addr_expr) = expr.kind;
//             if let ExprKind::Unary(UnOp::UnDeref, deref_expr) = addr_expr.kind;
//             if !in_external_macro(cx.sess(), expr.span);
//             if let Some(parent_expr) = get_parent_expr(cx, expr);
//             if match parent_expr.kind {
//                 ExprKind::Call(func, _) => func.hir_id != expr.hir_id,
//                 ExprKind::MethodCall(..) => true,
//                 _ => false,
//             };
//             if !cx.typeck_results().expr_ty(deref_expr).is_unsafe_ptr();
//             then {
//                 let mut app = Applicability::MachineApplicable;
//                 let sugg = format!("&{}", snippet_with_applicability(cx, deref_expr.span, "_", &mut app));
//                 span_lint_and_sugg(
//                     cx,
//                     REDUNDANT_DEREF,
//                     expr.span,
//                     "redundant dereference",
//                     "remove the dereference",
//                     sugg,
//                     app,
//                 );
//             }
//         }
//     }
// }
