use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{get_parent_expr, is_trait_method};
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to [`IntoIterator::into_iter`](https://doc.rust-lang.org/stable/std/iter/trait.IntoIterator.html#tymethod.into_iter)
    /// in a call argument that accepts `IntoIterator`.
    ///
    /// ### Why is this bad?
    /// If a generic parameter has an `IntoIterator` bound, there is no need to call `.into_iter()` at call site.
    /// Calling `IntoIterator::into_iter()` on a value implies that its type already implements `IntoIterator`,
    /// so you can just pass the value as is.
    ///
    /// ### Example
    /// ```rust
    /// fn even_sum<I: IntoIterator<Item = i32>>(iter: I) -> i32 {
    ///     iter.into_iter().filter(|&x| x % 2 == 0).sum()
    /// }
    ///
    /// let _ = even_sum([1, 2, 3].into_iter());
    /// //                        ^^^^^^^^^^^^ redundant. `[i32; 3]` implements `IntoIterator`
    /// ```
    /// Use instead:
    /// ```rust
    /// fn even_sum<I: IntoIterator<Item = i32>>(iter: I) -> i32 {
    ///     iter.into_iter().filter(|&x| x % 2 == 0).sum()
    /// }
    ///
    /// let _ = even_sum([1, 2, 3]);
    /// ```
    #[clippy::version = "1.71.0"]
    pub EXPLICIT_INTO_ITER_FN_ARG,
    pedantic,
    "explicit call to `.into_iter()` in function argument accepting `IntoIterator`"
}
declare_lint_pass!(ExplicitIntoIterFnArg => [EXPLICIT_INTO_ITER_FN_ARG]);

enum MethodOrFunction {
    Method,
    Function,
}

impl MethodOrFunction {
    /// Maps the argument position in `pos` to the parameter position.
    /// For methods, `self` is skipped.
    fn param_pos(self, pos: usize) -> usize {
        match self {
            MethodOrFunction::Method => pos + 1,
            MethodOrFunction::Function => pos,
        }
    }
}

/// Returns the span of the `IntoIterator` trait bound in the function pointed to by `fn_did`
fn into_iter_bound(cx: &LateContext<'_>, fn_did: DefId, into_iter_did: DefId, param_index: u32) -> Option<Span> {
    cx.tcx
        .predicates_of(fn_did)
        .predicates
        .iter()
        .find_map(|&(ref pred, span)| {
            if let ty::PredicateKind::Clause(ty::Clause::Trait(tr)) = pred.kind().skip_binder()
                && tr.def_id() == into_iter_did
                && tr.self_ty().is_param(param_index)
            {
                Some(span)
            } else {
                None
            }
        })
}

fn into_iter_call<'hir>(cx: &LateContext<'_>, expr: &'hir Expr<'hir>) -> Option<&'hir Expr<'hir>> {
    if let ExprKind::MethodCall(name, recv, _, _) = expr.kind
        && is_trait_method(cx, expr, sym::IntoIterator)
        && name.ident.name == sym::into_iter
    {
        Some(recv)
    } else {
        None
    }
}

impl<'tcx> LateLintPass<'tcx> for ExplicitIntoIterFnArg {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        if let Some(recv) = into_iter_call(cx, expr)
            && let Some(parent) = get_parent_expr(cx, expr)
            // Make sure that this is not a chained into_iter call (i.e. `x.into_iter().into_iter()`)
            // That case is already covered by `useless_conversion` and we don't want to lint twice
            // with two contradicting suggestions.
            && into_iter_call(cx, parent).is_none()
            && into_iter_call(cx, recv).is_none()
            && let Some(into_iter_did) = cx.tcx.get_diagnostic_item(sym::IntoIterator)
        {

            let parent = match parent.kind {
                ExprKind::Call(recv, args) if let ExprKind::Path(ref qpath) = recv.kind => {
                    cx.qpath_res(qpath, recv.hir_id).opt_def_id()
                        .map(|did| (did, args, MethodOrFunction::Function))
                }
                ExprKind::MethodCall(.., args, _) => {
                    cx.typeck_results().type_dependent_def_id(parent.hir_id)
                        .map(|did| (did, args, MethodOrFunction::Method))
                }
                _ => None,
            };

            if let Some((parent_fn_did, args, kind)) = parent
                && let sig = cx.tcx.fn_sig(parent_fn_did).skip_binder().skip_binder()
                && let Some(arg_pos) = args.iter().position(|x| x.hir_id == expr.hir_id)
                && let Some(&into_iter_param) = sig.inputs().get(kind.param_pos(arg_pos))
                && let ty::Param(param) = into_iter_param.kind()
                && let Some(span) = into_iter_bound(cx, parent_fn_did, into_iter_did, param.index)
            {
                let mut applicability = Applicability::MachineApplicable;
                let sugg = snippet_with_applicability(cx, recv.span.source_callsite(), "<expr>", &mut applicability).into_owned();

                span_lint_and_then(cx, EXPLICIT_INTO_ITER_FN_ARG, expr.span, "explicit call to `.into_iter()` in function argument accepting `IntoIterator`", |diag| {
                    diag.span_suggestion(
                        expr.span,
                        "consider removing `.into_iter()`",
                        sugg,
                        applicability,
                    );
                    diag.span_note(span, "this parameter accepts any `IntoIterator`, so you don't need to call `.into_iter()`");
                });
            }
        }
    }
}
