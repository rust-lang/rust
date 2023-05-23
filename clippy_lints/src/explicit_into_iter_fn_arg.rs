use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
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
    /// If a function has a generic parameter with an `IntoIterator` trait bound, it means that the function
    /// will *have* to call `.into_iter()` to get an iterator out of it, thereby making the call to `.into_iter()`
    /// at call site redundant.
    ///
    /// Consider this example:
    /// ```rs,ignore
    /// fn foo<T: IntoIterator<Item = u32>>(iter: T) {
    ///   let it = iter.into_iter();
    ///                ^^^^^^^^^^^^ the function has to call `.into_iter()` to get the iterator
    /// }
    ///
    /// foo(vec![1, 2, 3].into_iter());
    ///                  ^^^^^^^^^^^^ ... making this `.into_iter()` call redundant.
    /// ```
    ///
    /// The reason for why calling `.into_iter()` twice (once at call site and again inside of the function) works in the first place
    /// is because there is a blanket implementation of `IntoIterator` for all types that implement `Iterator` in the standard library,
    /// in which it simply returns itself, effectively making the second call to `.into_iter()` a "no-op":
    /// ```rust,ignore
    /// impl<I: Iterator> IntoIterator for I {
    ///     type Item = I::Item;
    ///     type IntoIter = I;
    ///
    ///     fn into_iter(self) -> I {
    ///        self
    ///     }
    /// }
    /// ```
    ///
    /// ### Example
    /// ```rust
    /// fn even_sum<I: IntoIterator<Item = u32>>(iter: I) -> u32 {
    ///     iter.into_iter().filter(|&x| x % 2 == 0).sum()
    /// }
    ///
    /// let _ = even_sum(vec![1, 2, 3].into_iter());
    /// ```
    /// Use instead:
    /// ```rust
    /// fn even_sum<I: IntoIterator<Item = u32>>(iter: I) -> u32 {
    ///     iter.into_iter().filter(|&x| x % 2 == 0).sum()
    /// }
    ///
    /// let _ = even_sum(vec![1, 2, 3]);
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

/// Returns the span of the `IntoIterator` trait bound in the function pointed to by `fn_did`
fn into_iter_bound(cx: &LateContext<'_>, fn_did: DefId, param_index: u32) -> Option<Span> {
    if let Some(into_iter_did) = cx.tcx.get_diagnostic_item(sym::IntoIterator) {
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
    } else {
        None
    }
}

impl<'tcx> LateLintPass<'tcx> for ExplicitIntoIterFnArg {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        if let ExprKind::MethodCall(name, recv, ..) = expr.kind
            && is_trait_method(cx, expr, sym::IntoIterator)
            && name.ident.name == sym::into_iter
            && let Some(parent_expr) = get_parent_expr(cx, expr)
        {
            let parent_expr = match parent_expr.kind {
                ExprKind::Call(recv, args) if let ExprKind::Path(ref qpath) = recv.kind => {
                    cx.qpath_res(qpath, recv.hir_id).opt_def_id()
                        .map(|did| (did, args, MethodOrFunction::Function))
                }
                ExprKind::MethodCall(.., args, _) => {
                    cx.typeck_results().type_dependent_def_id(parent_expr.hir_id)
                        .map(|did| (did, args, MethodOrFunction::Method))
                }
                _ => None,
            };

            if let Some((parent_fn_did, args, kind)) = parent_expr
                && let sig = cx.tcx.fn_sig(parent_fn_did).skip_binder().skip_binder()
                && let Some(arg_pos) = args.iter().position(|x| x.hir_id == expr.hir_id)
                && let Some(&into_iter_param) = sig.inputs().get(match kind {
                    MethodOrFunction::Function => arg_pos,
                    MethodOrFunction::Method => arg_pos + 1, // skip self arg
                })
                && let ty::Param(param) = into_iter_param.kind()
                && let Some(span) = into_iter_bound(cx, parent_fn_did, param.index)
            {
                let sugg = snippet(cx, recv.span.source_callsite(), "<expr>").into_owned();
                span_lint_and_then(cx, EXPLICIT_INTO_ITER_FN_ARG, expr.span, "explicit call to `.into_iter()` in function argument accepting `IntoIterator`", |diag| {
                    diag.span_suggestion(
                        expr.span,
                        "consider removing `.into_iter()`",
                        sugg,
                        Applicability::MachineApplicable,
                    );
                    diag.span_note(span, "this parameter accepts any `IntoIterator`, so you don't need to call `.into_iter()`");
                });
            }
        }
    }
}
