use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::{find_assert_eq_args, root_macro_call_first_node};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{implements_trait, is_copy};
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Lit};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{self, Ty};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::Ident;

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns about boolean comparisons in assert-like macros.
    ///
    /// ### Why is this bad?
    /// It is shorter to use the equivalent.
    ///
    /// ### Example
    /// ```no_run
    /// assert_eq!("a".is_empty(), false);
    /// assert_ne!("a".is_empty(), true);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// assert!(!"a".is_empty());
    /// ```
    #[clippy::version = "1.53.0"]
    pub BOOL_ASSERT_COMPARISON,
    style,
    "Using a boolean as comparison value in an assert_* macro when there is no need"
}

declare_lint_pass!(BoolAssertComparison => [BOOL_ASSERT_COMPARISON]);

fn extract_bool_lit(e: &Expr<'_>) -> Option<bool> {
    if let ExprKind::Lit(Lit {
        node: LitKind::Bool(b), ..
    }) = e.kind
        && !e.span.from_expansion()
    {
        Some(*b)
    } else {
        None
    }
}

fn is_impl_not_trait_with_bool_out<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
    cx.tcx
        .lang_items()
        .not_trait()
        .filter(|trait_id| implements_trait(cx, ty, *trait_id, &[]))
        .and_then(|trait_id| {
            cx.tcx.associated_items(trait_id).find_by_ident_and_kind(
                cx.tcx,
                Ident::from_str("Output"),
                ty::AssocTag::Type,
                trait_id,
            )
        })
        .is_some_and(|assoc_item| {
            let proj = Ty::new_projection(cx.tcx, assoc_item.def_id, cx.tcx.mk_args_trait(ty, []));
            let nty = cx.tcx.normalize_erasing_regions(cx.typing_env(), proj);

            nty.is_bool()
        })
}

impl<'tcx> LateLintPass<'tcx> for BoolAssertComparison {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let Some(macro_call) = root_macro_call_first_node(cx, expr) else {
            return;
        };
        let macro_name = cx.tcx.item_name(macro_call.def_id);
        let eq_macro = match macro_name.as_str() {
            "assert_eq" | "debug_assert_eq" => true,
            "assert_ne" | "debug_assert_ne" => false,
            _ => return,
        };
        let Some((a, b, _)) = find_assert_eq_args(cx, expr, macro_call.expn) else {
            return;
        };

        let a_span = a.span.source_callsite();
        let b_span = b.span.source_callsite();

        let (lit_span, bool_value, non_lit_expr) = match (extract_bool_lit(a), extract_bool_lit(b)) {
            // assert_eq!(true/false, b)
            //            ^^^^^^^^^^^^
            (Some(bool_value), None) => (a_span.until(b_span), bool_value, b),
            // assert_eq!(a, true/false)
            //             ^^^^^^^^^^^^
            (None, Some(bool_value)) => (b_span.with_lo(a_span.hi()), bool_value, a),
            // If there are two boolean arguments, we definitely don't understand
            // what's going on, so better leave things as is...
            //
            // Or there is simply no boolean and then we can leave things as is!
            _ => return,
        };

        let non_lit_ty = cx.typeck_results().expr_ty(non_lit_expr);

        if !is_impl_not_trait_with_bool_out(cx, non_lit_ty) {
            // At this point the expression which is not a boolean
            // literal does not implement Not trait with a bool output,
            // so we cannot suggest to rewrite our code
            return;
        }

        if !is_copy(cx, non_lit_ty) {
            // Only lint with types that are `Copy` because `assert!(x)` takes
            // ownership of `x` whereas `assert_eq(x, true)` does not
            return;
        }

        let macro_name = macro_name.as_str();
        let non_eq_mac = &macro_name[..macro_name.len() - 3];
        span_lint_and_then(
            cx,
            BOOL_ASSERT_COMPARISON,
            macro_call.span,
            format!("used `{macro_name}!` with a literal bool"),
            |diag| {
                // assert_eq!(...)
                // ^^^^^^^^^
                let name_span = cx.sess().source_map().span_until_char(macro_call.span, '!');

                let mut suggestions = vec![(name_span, non_eq_mac.to_string()), (lit_span, String::new())];

                if bool_value ^ eq_macro {
                    let Some(sugg) = Sugg::hir_opt(cx, non_lit_expr) else {
                        return;
                    };
                    suggestions.push((non_lit_expr.span, (!sugg).to_string()));
                }

                diag.multipart_suggestion(
                    format!("replace it with `{non_eq_mac}!(..)`"),
                    suggestions,
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}
