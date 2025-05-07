use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::visitors::find_all_ret_expressions;
use clippy_utils::{contains_return, is_res_lang_ctor, path_res, return_ty};
use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionSome, ResultOk};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, ExprKind, FnDecl, Impl, ItemKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::sym;

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
    /// ```no_run
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
    /// ```no_run
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
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            avoid_breaking_exported_api: conf.avoid_breaking_exported_api,
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
        def_id: LocalDefId,
    ) {
        // Abort if public function/method or closure.
        match fn_kind {
            FnKind::ItemFn(..) | FnKind::Method(..) => {
                if self.avoid_breaking_exported_api && cx.effective_visibilities.is_exported(def_id) {
                    return;
                }
            },
            FnKind::Closure => return,
        }

        // Abort if the method is implementing a trait or of it a trait method.
        let hir_id = cx.tcx.local_def_id_to_hir_id(def_id);
        if let Node::Item(item) = cx.tcx.parent_hir_node(hir_id)
            && matches!(
                item.kind,
                ItemKind::Impl(Impl { of_trait: Some(_), .. }) | ItemKind::Trait(..)
            )
        {
            return;
        }

        // Get the wrapper and inner types, if can't, abort.
        let (return_type_label, lang_item, inner_type) =
            if let ty::Adt(adt_def, subst) = return_ty(cx, hir_id.expect_owner()).kind() {
                if cx.tcx.is_diagnostic_item(sym::Option, adt_def.did()) {
                    ("Option", OptionSome, subst.type_at(0))
                } else if cx.tcx.is_diagnostic_item(sym::Result, adt_def.did()) {
                    ("Result", ResultOk, subst.type_at(0))
                } else {
                    return;
                }
            } else {
                return;
            };

        // Check if all return expression respect the following condition and collect them.
        let mut suggs = Vec::new();
        let can_sugg = find_all_ret_expressions(cx, body.value, |ret_expr| {
            if !ret_expr.span.from_expansion()
                // Check if a function call.
                && let ExprKind::Call(func, [arg]) = ret_expr.kind
                && is_res_lang_ctor(cx, path_res(cx, func), lang_item)
                // Make sure the function argument does not contain a return expression.
                && !contains_return(arg)
            {
                suggs.push((
                    ret_expr.span,
                    if inner_type.is_unit() {
                        String::new()
                    } else {
                        snippet(cx, arg.span.source_callsite(), "..").to_string()
                    },
                ));
                true
            } else {
                false
            }
        });

        if can_sugg && !suggs.is_empty() {
            let (lint_msg, return_type_sugg_msg, return_type_sugg, body_sugg_msg) = if inner_type.is_unit() {
                (
                    "this function's return value is unnecessary".to_string(),
                    "remove the return type...".to_string(),
                    // FIXME: we should instead get the span including the `->` and suggest an
                    // empty string for this case.
                    "()".to_string(),
                    "...and then remove returned values",
                )
            } else {
                (
                    format!("this function's return value is unnecessarily wrapped by `{return_type_label}`"),
                    format!("remove `{return_type_label}` from the return type..."),
                    inner_type.to_string(),
                    "...and then change returning expressions",
                )
            };

            span_lint_and_then(cx, UNNECESSARY_WRAPS, span, lint_msg, |diag| {
                diag.span_suggestion(
                    fn_decl.output.span(),
                    return_type_sugg_msg,
                    return_type_sugg,
                    Applicability::MaybeIncorrect,
                );
                diag.multipart_suggestion(body_sugg_msg, suggs, Applicability::MaybeIncorrect);
            });
        }
    }
}
