use crate::utils::{
    contains_return, in_macro, is_type_diagnostic_item, match_qpath, paths, return_ty, snippet, span_lint_and_then,
    visitors::find_all_ret_expressions,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, ExprKind, FnDecl, HirId, Impl, ItemKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::subst::GenericArgKind;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::sym;
use rustc_span::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for private functions that only return `Ok` or `Some`.
    ///
    /// **Why is this bad?** It is not meaningful to wrap values when no `None` or `Err` is returned.
    ///
    /// **Known problems:** Since this lint changes function type signature, you may need to
    /// adjust some code at callee side.
    ///
    /// **Example:**
    ///
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
    pub UNNECESSARY_WRAPS,
    complexity,
    "functions that only return `Ok` or `Some`"
}

declare_lint_pass!(UnnecessaryWraps => [UNNECESSARY_WRAPS]);

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
        match fn_kind {
            FnKind::ItemFn(.., visibility, _) | FnKind::Method(.., Some(visibility), _) => {
                if visibility.node.is_pub() {
                    return;
                }
            },
            FnKind::Closure(..) => return,
            _ => (),
        }

        if let Some(Node::Item(item)) = cx.tcx.hir().find(cx.tcx.hir().get_parent_node(hir_id)) {
            if matches!(
                item.kind,
                ItemKind::Impl(Impl { of_trait: Some(_), .. }) | ItemKind::Trait(..)
            ) {
                return;
            }
        }

        let (return_type, path) = if is_type_diagnostic_item(cx, return_ty(cx, hir_id), sym::option_type) {
            ("Option", &paths::OPTION_SOME)
        } else if is_type_diagnostic_item(cx, return_ty(cx, hir_id), sym::result_type) {
            ("Result", &paths::RESULT_OK)
        } else {
            return;
        };

        let mut suggs = Vec::new();
        let can_sugg = find_all_ret_expressions(cx, &body.value, |ret_expr| {
            if_chain! {
                if !in_macro(ret_expr.span);
                if let ExprKind::Call(ref func, ref args) = ret_expr.kind;
                if let ExprKind::Path(ref qpath) = func.kind;
                if match_qpath(qpath, path);
                if args.len() == 1;
                if !contains_return(&args[0]);
                then {
                    suggs.push((ret_expr.span, snippet(cx, args[0].span.source_callsite(), "..").to_string()));
                    true
                } else {
                    false
                }
            }
        });

        if can_sugg && !suggs.is_empty() {
            span_lint_and_then(
                cx,
                UNNECESSARY_WRAPS,
                span,
                format!(
                    "this function's return value is unnecessarily wrapped by `{}`",
                    return_type
                )
                .as_str(),
                |diag| {
                    let inner_ty = return_ty(cx, hir_id)
                        .walk()
                        .skip(1) // skip `std::option::Option` or `std::result::Result`
                        .take(1) // take the first outermost inner type
                        .filter_map(|inner| match inner.unpack() {
                            GenericArgKind::Type(inner_ty) => Some(inner_ty.to_string()),
                            _ => None,
                        });
                    inner_ty.for_each(|inner_ty| {
                        diag.span_suggestion(
                            fn_decl.output.span(),
                            format!("remove `{}` from the return type...", return_type).as_str(),
                            inner_ty,
                            Applicability::MaybeIncorrect,
                        );
                    });
                    diag.multipart_suggestion(
                        "...and change the returning expressions",
                        suggs,
                        Applicability::MaybeIncorrect,
                    );
                },
            );
        }
    }
}
