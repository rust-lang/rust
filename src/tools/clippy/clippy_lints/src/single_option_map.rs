use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{path_res, peel_blocks};
use rustc_hir::def::Res;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, ExprKind, FnDecl, FnRetTy};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions with method calls to `.map(_)` on an arg
    /// of type `Option` as the outermost expression.
    ///
    /// ### Why is this bad?
    /// Taking and returning an `Option<T>` may require additional
    /// `Some(_)` and `unwrap` if all you have is a `T`.
    ///
    /// ### Example
    /// ```no_run
    /// fn double(param: Option<u32>) -> Option<u32> {
    ///     param.map(|x| x * 2)
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn double(param: u32) -> u32 {
    ///     param * 2
    /// }
    /// ```
    #[clippy::version = "1.87.0"]
    pub SINGLE_OPTION_MAP,
    nursery,
    "Checks for functions with method calls to `.map(_)` on an arg of type `Option` as the outermost expression."
}

declare_lint_pass!(SingleOptionMap => [SINGLE_OPTION_MAP]);

impl<'tcx> LateLintPass<'tcx> for SingleOptionMap {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'tcx>,
        body: &'tcx Body<'tcx>,
        span: Span,
        _fn_def: LocalDefId,
    ) {
        if let FnRetTy::Return(_ret) = decl.output
            && matches!(kind, FnKind::ItemFn(_, _, _) | FnKind::Method(_, _))
        {
            let func_body = peel_blocks(body.value);
            if let ExprKind::MethodCall(method_name, callee, args, _span) = func_body.kind
                && method_name.ident.name == sym::map
                && let callee_type = cx.typeck_results().expr_ty(callee)
                && is_type_diagnostic_item(cx, callee_type, sym::Option)
                && let ExprKind::Path(_path) = callee.kind
                && let Res::Local(_id) = path_res(cx, callee)
                && matches!(path_res(cx, callee), Res::Local(_id))
                && !matches!(args[0].kind, ExprKind::Path(_))
            {
                if let ExprKind::Closure(closure) = args[0].kind {
                    let Body { params: [..], value } = cx.tcx.hir_body(closure.body);
                    if let ExprKind::Call(func, f_args) = value.kind
                        && matches!(func.kind, ExprKind::Path(_))
                        && f_args.iter().all(|arg| matches!(arg.kind, ExprKind::Path(_)))
                    {
                        return;
                    } else if let ExprKind::MethodCall(_segment, receiver, method_args, _span) = value.kind
                        && matches!(receiver.kind, ExprKind::Path(_))
                        && method_args.iter().all(|arg| matches!(arg.kind, ExprKind::Path(_)))
                        && method_args.iter().all(|arg| matches!(path_res(cx, arg), Res::Local(_)))
                    {
                        return;
                    }
                }

                span_lint_and_help(
                    cx,
                    SINGLE_OPTION_MAP,
                    span,
                    "`fn` that only maps over argument",
                    None,
                    "move the `.map` to the caller or to an `_opt` function",
                );
            }
        }
    }
}
