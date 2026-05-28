use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{SpanRangeExt, position_before_rarrow, snippet_block};
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{
    Block, Body, Closure, ClosureKind, CoroutineDesugaring, CoroutineKind, CoroutineSource, Expr, ExprKind, FnDecl,
    FnRetTy, GenericBound, Node, OpaqueTy, TraitRef, Ty, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::middle::resolve_bound_vars::ResolvedArg;
use rustc_middle::ty;
use rustc_session::declare_lint_pass;
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// It checks for manual implementations of `async` functions.
    ///
    /// ### Why is this bad?
    /// It's more idiomatic to use the dedicated syntax.
    ///
    /// ### Example
    /// ```no_run
    /// use std::future::Future;
    ///
    /// fn foo() -> impl Future<Output = i32> { async { 42 } }
    /// ```
    /// Use instead:
    /// ```no_run
    /// async fn foo() -> i32 { 42 }
    /// ```
    #[clippy::version = "1.45.0"]
    pub MANUAL_ASYNC_FN,
    style,
    "manual implementations of `async` functions can be simplified using the dedicated syntax"
}

declare_lint_pass!(ManualAsyncFn => [MANUAL_ASYNC_FN]);

impl<'tcx> LateLintPass<'tcx> for ManualAsyncFn {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        span: Span,
        fn_def_id: LocalDefId,
    ) {
        if let Some(header) = kind.header()
            && !header.asyncness.is_async()
            // Check that this function returns `impl Future`
            && let FnRetTy::Return(ret_ty) = decl.output
            && let TyKind::OpaqueDef(opaque) = ret_ty.kind
            && let Some(trait_ref) = future_trait_ref(cx, opaque)
            && let Some(output) = future_output_ty(trait_ref)
            && captures_all_lifetimes(cx, fn_def_id, opaque.def_id)
            // Check that the body of the function consists of one async block
            && let ExprKind::Block(block, _) = body.value.kind
            && block.stmts.is_empty()
            && let Some(closure_body) = desugared_async_block(cx, block)
            && let Some(vis_span_opt) = match cx.tcx.hir_node_by_def_id(fn_def_id) {
                Node::Item(item) => Some(Some(item.vis_span)),
                Node::ImplItem(impl_item) => Some(impl_item.vis_span()),
                _ => None,
            }
            && !span.from_expansion()
        {
            let header_span = span.with_hi(ret_ty.span.hi());

            span_lint_and_then(
                cx,
                MANUAL_ASYNC_FN,
                header_span,
                "this function can be simplified using the `async fn` syntax",
                |diag| {
                    if let Some(vis_span) = vis_span_opt
                        && let Some(vis_snip) = vis_span.get_source_text(cx)
                        && let Some(header_snip) = header_span.get_source_text(cx)
                        && let Some(ret_pos) = position_before_rarrow(&header_snip)
                        && let Some((_, ret_snip)) = suggested_ret(cx, output)
                    {
                        let header_snip = if vis_snip.is_empty() {
                            format!("async {}", &header_snip[..ret_pos])
                        } else {
                            format!("{} async {}", vis_snip, &header_snip[vis_snip.len() + 1..ret_pos])
                        };

                        let body_snip = snippet_block(cx, closure_body.value.span, "..", Some(block.span));

                        diag.multipart_suggestion(
                            "make the function `async` and return the output of the future directly",
                            vec![
                                (header_span, format!("{header_snip}{ret_snip}")),
                                (block.span, body_snip),
                            ],
                            Applicability::MachineApplicable,
                        );
                    }
                },
            );
        }
    }
}

fn future_trait_ref<'tcx>(cx: &LateContext<'tcx>, opaque: &'tcx OpaqueTy<'tcx>) -> Option<&'tcx TraitRef<'tcx>> {
    if let Some(trait_ref) = opaque.bounds.iter().find_map(|bound| {
        if let GenericBound::Trait(poly) = bound {
            Some(&poly.trait_ref)
        } else {
            None
        }
    }) && trait_ref.trait_def_id() == cx.tcx.lang_items().future_trait()
    {
        return Some(trait_ref);
    }

    None
}

fn future_output_ty<'tcx>(trait_ref: &'tcx TraitRef<'tcx>) -> Option<&'tcx Ty<'tcx>> {
    if let Some(segment) = trait_ref.path.segments.last()
        && let Some(args) = segment.args
        && let [constraint] = args.constraints
        && constraint.ident.name == sym::Output
        && let Some(output) = constraint.ty()
    {
        return Some(output);
    }

    None
}

fn captures_all_lifetimes(cx: &LateContext<'_>, fn_def_id: LocalDefId, opaque_def_id: LocalDefId) -> bool {
    let early_input_params = ty::GenericArgs::identity_for_item(cx.tcx, fn_def_id);
    let late_input_params = cx.tcx.late_bound_vars(cx.tcx.local_def_id_to_hir_id(fn_def_id));

    let num_early_lifetimes = early_input_params
        .iter()
        .filter(|param| param.as_region().is_some())
        .count();
    let num_late_lifetimes = late_input_params
        .iter()
        .filter(|param_kind| matches!(param_kind, ty::BoundVariableKind::Region(_)))
        .count();

    // There is no lifetime, so they are all captured.
    if num_early_lifetimes == 0 && num_late_lifetimes == 0 {
        return true;
    }

    // By construction, each captured lifetime only appears once in `opaque_captured_lifetimes`.
    let num_captured_lifetimes = cx
        .tcx
        .opaque_captured_lifetimes(opaque_def_id)
        .iter()
        .filter(|&(lifetime, _)| {
            matches!(
                *lifetime,
                ResolvedArg::EarlyBound(_) | ResolvedArg::LateBound(ty::INNERMOST, _, _)
            )
        })
        .count();
    num_captured_lifetimes == num_early_lifetimes + num_late_lifetimes
}

fn desugared_async_block<'tcx>(cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) -> Option<&'tcx Body<'tcx>> {
    if let Some(&Expr {
        kind: ExprKind::Closure(&Closure { kind, body, .. }),
        ..
    }) = block.expr
        && let ClosureKind::Coroutine(CoroutineKind::Desugared(CoroutineDesugaring::Async, CoroutineSource::Block)) =
            kind
    {
        return Some(cx.tcx.hir_body(body));
    }

    None
}

fn suggested_ret(cx: &LateContext<'_>, output: &Ty<'_>) -> Option<(&'static str, String)> {
    if let TyKind::Tup([]) = output.kind {
        let sugg = "remove the return type";
        Some((sugg, String::new()))
    } else {
        let sugg = "return the output of the future directly";
        output.span.get_source_text(cx).map(|src| (sugg, format!(" -> {src}")))
    }
}
