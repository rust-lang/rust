use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{return_ty, sym};
use rustc_hir::{
    Body, BodyOwnerKind, Expr, ExprKind, FnSig, ImplItem, ImplItemKind, Item, ItemKind, OwnerId, PathSegment, QPath,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::impl_lint_pass;
use rustc_span::{Ident, Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions of type `Result` that contain `expect()` or `unwrap()`
    ///
    /// ### Why restrict this?
    /// These functions promote recoverable errors to non-recoverable errors,
    /// which may be undesirable in code bases which wish to avoid panics,
    /// or be a bug in the specific function.
    ///
    /// ### Known problems
    /// This can cause false positives in functions that handle both recoverable and non recoverable errors.
    ///
    /// ### Example
    /// Before:
    /// ```no_run
    /// fn divisible_by_3(i_str: String) -> Result<(), String> {
    ///     let i = i_str
    ///         .parse::<i32>()
    ///         .expect("cannot divide the input by three");
    ///
    ///     if i % 3 != 0 {
    ///         Err("Number is not divisible by 3")?
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    ///
    /// After:
    /// ```no_run
    /// fn divisible_by_3(i_str: String) -> Result<(), String> {
    ///     let i = i_str
    ///         .parse::<i32>()
    ///         .map_err(|e| format!("cannot divide the input by three: {}", e))?;
    ///
    ///     if i % 3 != 0 {
    ///         Err("Number is not divisible by 3")?
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    #[clippy::version = "1.48.0"]
    pub UNWRAP_IN_RESULT,
    restriction,
    "functions of type `Result<..>` or `Option`<...> that contain `expect()` or `unwrap()`"
}

impl_lint_pass!(UnwrapInResult=> [UNWRAP_IN_RESULT]);

#[derive(Clone, Copy, Eq, PartialEq)]
enum OptionOrResult {
    Option,
    Result,
}

impl OptionOrResult {
    fn with_article(self) -> &'static str {
        match self {
            Self::Option => "an `Option`",
            Self::Result => "a `Result`",
        }
    }
}

struct OptionOrResultFn {
    kind: OptionOrResult,
    return_ty_span: Option<Span>,
}

#[derive(Default)]
pub struct UnwrapInResult {
    fn_stack: Vec<Option<OptionOrResultFn>>,
    current_fn: Option<OptionOrResultFn>,
}

impl UnwrapInResult {
    fn enter_item(&mut self, cx: &LateContext<'_>, fn_def_id: OwnerId, sig: &FnSig<'_>) {
        self.fn_stack.push(self.current_fn.take());
        self.current_fn = is_option_or_result(cx, return_ty(cx, fn_def_id)).map(|kind| OptionOrResultFn {
            kind,
            return_ty_span: Some(sig.decl.output.span()),
        });
    }

    fn leave_item(&mut self) {
        self.current_fn = self.fn_stack.pop().unwrap();
    }
}

impl<'tcx> LateLintPass<'tcx> for UnwrapInResult {
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx ImplItem<'_>) {
        if let ImplItemKind::Fn(sig, _) = &impl_item.kind {
            self.enter_item(cx, impl_item.owner_id, sig);
        }
    }

    fn check_impl_item_post(&mut self, _: &LateContext<'tcx>, impl_item: &'tcx ImplItem<'tcx>) {
        if let ImplItemKind::Fn(..) = impl_item.kind {
            self.leave_item();
        }
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Fn {
            has_body: true, sig, ..
        } = &item.kind
        {
            self.enter_item(cx, item.owner_id, sig);
        }
    }

    fn check_item_post(&mut self, _: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Fn { has_body: true, .. } = item.kind {
            self.leave_item();
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        if let Some(OptionOrResultFn {
            kind,
            ref mut return_ty_span,
        }) = self.current_fn
            && let Some((oor, fn_name)) = is_unwrap_or_expect_call(cx, expr)
            && oor == kind
        {
            span_lint_and_then(
                cx,
                UNWRAP_IN_RESULT,
                expr.span,
                format!("`{fn_name}` used in a function that returns {}", kind.with_article()),
                |diag| {
                    // Issue the note and help only once per function
                    if let Some(span) = return_ty_span.take() {
                        diag.span_note(span, "in this function signature");
                        let complement = if kind == OptionOrResult::Result {
                            " or calling the `.map_err()` method"
                        } else {
                            ""
                        };
                        diag.help(format!("consider using the `?` operator{complement}"));
                    }
                },
            );
        }
    }

    fn check_body(&mut self, cx: &LateContext<'tcx>, body: &Body<'tcx>) {
        let body_def_id = cx.tcx.hir_body_owner_def_id(body.id());
        if !matches!(cx.tcx.hir_body_owner_kind(body_def_id), BodyOwnerKind::Fn) {
            // When entering a body which is not a function, mask the potential surrounding
            // function to not apply the lint.
            self.fn_stack.push(self.current_fn.take());
        }
    }

    fn check_body_post(&mut self, cx: &LateContext<'tcx>, body: &Body<'tcx>) {
        let body_def_id = cx.tcx.hir_body_owner_def_id(body.id());
        if !matches!(cx.tcx.hir_body_owner_kind(body_def_id), BodyOwnerKind::Fn) {
            // Unmask the potential surrounding function.
            self.current_fn = self.fn_stack.pop().unwrap();
        }
    }
}

fn is_option_or_result(cx: &LateContext<'_>, ty: Ty<'_>) -> Option<OptionOrResult> {
    match ty.ty_adt_def().and_then(|def| cx.tcx.get_diagnostic_name(def.did())) {
        Some(sym::Option) => Some(OptionOrResult::Option),
        Some(sym::Result) => Some(OptionOrResult::Result),
        _ => None,
    }
}

fn is_unwrap_or_expect_call(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<(OptionOrResult, Symbol)> {
    if let ExprKind::Call(func, _) = expr.kind
        && let ExprKind::Path(QPath::TypeRelative(
            hir_ty,
            PathSegment {
                ident:
                    Ident {
                        name: name @ (sym::unwrap | sym::expect),
                        ..
                    },
                ..
            },
        )) = func.kind
    {
        is_option_or_result(cx, cx.typeck_results().node_type(hir_ty.hir_id)).map(|oor| (oor, *name))
    } else if let ExprKind::MethodCall(
        PathSegment {
            ident: Ident {
                name: name @ (sym::unwrap | sym::expect),
                ..
            },
            ..
        },
        recv,
        _,
        _,
    ) = expr.kind
    {
        is_option_or_result(cx, cx.typeck_results().expr_ty_adjusted(recv)).map(|oor| (oor, *name))
    } else {
        None
    }
}
