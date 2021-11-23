use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::{contains_return, is_lang_ctor, return_ty, visitors::find_all_ret_expressions};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::LangItem::{OptionSome, ResultOk};
use rustc_hir::{Body, ExprKind, FnDecl, HirId, Impl, ItemKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::sym;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for private functions that only return `Ok` or `Some`.
    ///
    /// ### Why is this bad?
    /// It is not meaningful to wrap values when no `None` or `Err` is returned.
    ///
    /// ### Known problems
    /// There can be false positives if the function signature is designed to
    /// fit some external requirement.
    ///
    /// ### Example
    /// ```rust
    /// fn get_cool_number(a: bool, b: bool) -> Option<i32> {
    ///     if a && b {
    ///         return Some(50);
    ///     }
    ///     if a {
    ///         Some(0)
    ///     } else {
    ///         Some(10)
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn get_cool_number(a: bool, b: bool) -> i32 {
    ///     if a && b {
    ///         return 50;
    ///     }
    ///     if a {
    ///         0
    ///     } else {
    ///         10
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.50.0"]
    pub UNNECESSARY_WRAPS,
    pedantic,
    "functions that only return `Ok` or `Some`"
}

pub struct UnnecessaryWraps {
    avoid_breaking_exported_api: bool,
}

impl_lint_pass!(UnnecessaryWraps => [UNNECESSARY_WRAPS]);

impl UnnecessaryWraps {
    pub fn new(avoid_breaking_exported_api: bool) -> Self {
        Self {
            avoid_breaking_exported_api,
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for UnnecessaryWraps {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'tcx>,
        fn_decl: &FnDecl<'tcx>,
        body: &Body<'tcx>,
        span: Span,
        hir_id: HirId,
    ) {
        // Abort if public function/method or closure.
        match fn_kind {
            FnKind::ItemFn(..) | FnKind::Method(..) => {
                let def_id = cx.tcx.hir().local_def_id(hir_id);
                if self.avoid_breaking_exported_api && cx.access_levels.is_exported(def_id) {
                    return;
                }
            },
            FnKind::Closure => return,
        }

        // Abort if the method is implementing a trait or of it a trait method.
        if let Some(Node::Item(item)) = cx.tcx.hir().find(cx.tcx.hir().get_parent_node(hir_id)) {
            if matches!(
                item.kind,
                ItemKind::Impl(Impl { of_trait: Some(_), .. }) | ItemKind::Trait(..)
            ) {
                return;
            }
        }

        // Get the wrapper and inner types, if can't, abort.
        let (return_type_label, lang_item, inner_type) = if let ty::Adt(adt_def, subst) = return_ty(cx, hir_id).kind() {
            if cx.tcx.is_diagnostic_item(sym::Option, adt_def.did) {
                ("Option", OptionSome, subst.type_at(0))
            } else if cx.tcx.is_diagnostic_item(sym::Result, adt_def.did) {
                ("Result", ResultOk, subst.type_at(0))
            } else {
                return;
            }
        } else {
            return;
        };

        // Check if all return expression respect the following condition and collect them.
        let mut suggs = Vec::new();
        let can_sugg = find_all_ret_expressions(cx, &body.value, |ret_expr| {
            if_chain! {
                if !ret_expr.span.from_expansion();
                // Check if a function call.
                if let ExprKind::Call(func, [arg]) = ret_expr.kind;
                // Check if OPTION_SOME or RESULT_OK, depending on return type.
                if let ExprKind::Path(qpath) = &func.kind;
                if is_lang_ctor(cx, qpath, lang_item);
                // Make sure the function argument does not contain a return expression.
                if !contains_return(arg);
                then {
                    suggs.push(
                        (
                            ret_expr.span,
                            if inner_type.is_unit() {
                                "".to_string()
                            } else {
                                snippet(cx, arg.span.source_callsite(), "..").to_string()
                            }
                        )
                    );
                    true
                } else {
                    false
                }
            }
        });

        if can_sugg && !suggs.is_empty() {
            let (lint_msg, return_type_sugg_msg, return_type_sugg, body_sugg_msg) = if inner_type.is_unit() {
                (
                    "this function's return value is unnecessary".to_string(),
                    "remove the return type...".to_string(),
                    snippet(cx, fn_decl.output.span(), "..").to_string(),
                    "...and then remove returned values",
                )
            } else {
                (
                    format!(
                        "this function's return value is unnecessarily wrapped by `{}`",
                        return_type_label
                    ),
                    format!("remove `{}` from the return type...", return_type_label),
                    inner_type.to_string(),
                    "...and then change returning expressions",
                )
            };

            span_lint_and_then(cx, UNNECESSARY_WRAPS, span, lint_msg.as_str(), |diag| {
                diag.span_suggestion(
                    fn_decl.output.span(),
                    return_type_sugg_msg.as_str(),
                    return_type_sugg,
                    Applicability::MaybeIncorrect,
                );
                diag.multipart_suggestion(body_sugg_msg, suggs, Applicability::MaybeIncorrect);
            });
        }
    }
}
