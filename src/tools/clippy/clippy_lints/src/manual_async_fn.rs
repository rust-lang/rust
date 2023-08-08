use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{position_before_rarrow, snippet_block, snippet_opt};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{
    AsyncGeneratorKind, Block, Body, Closure, Expr, ExprKind, FnDecl, FnRetTy, GeneratorKind, GenericArg, GenericBound,
    ImplItem, Item, ItemKind, LifetimeName, Node, Term, TraitRef, Ty, TyKind, TypeBindingKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::def_id::LocalDefId;
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// It checks for manual implementations of `async` functions.
    ///
    /// ### Why is this bad?
    /// It's more idiomatic to use the dedicated syntax.
    ///
    /// ### Example
    /// ```rust
    /// use std::future::Future;
    ///
    /// fn foo() -> impl Future<Output = i32> { async { 42 } }
    /// ```
    /// Use instead:
    /// ```rust
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
        def_id: LocalDefId,
    ) {
        if_chain! {
            if let Some(header) = kind.header();
            if !header.asyncness.is_async();
            // Check that this function returns `impl Future`
            if let FnRetTy::Return(ret_ty) = decl.output;
            if let Some((trait_ref, output_lifetimes)) = future_trait_ref(cx, ret_ty);
            if let Some(output) = future_output_ty(trait_ref);
            if captures_all_lifetimes(decl.inputs, &output_lifetimes);
            // Check that the body of the function consists of one async block
            if let ExprKind::Block(block, _) = body.value.kind;
            if block.stmts.is_empty();
            if let Some(closure_body) = desugared_async_block(cx, block);
            if let Node::Item(Item {vis_span, ..}) | Node::ImplItem(ImplItem {vis_span, ..}) =
                cx.tcx.hir().get_by_def_id(def_id);
            then {
                let header_span = span.with_hi(ret_ty.span.hi());

                span_lint_and_then(
                    cx,
                    MANUAL_ASYNC_FN,
                    header_span,
                    "this function can be simplified using the `async fn` syntax",
                    |diag| {
                        if_chain! {
                            if let Some(vis_snip) = snippet_opt(cx, *vis_span);
                            if let Some(header_snip) = snippet_opt(cx, header_span);
                            if let Some(ret_pos) = position_before_rarrow(&header_snip);
                            if let Some((ret_sugg, ret_snip)) = suggested_ret(cx, output);
                            then {
                                let header_snip = if vis_snip.is_empty() {
                                    format!("async {}", &header_snip[..ret_pos])
                                } else {
                                    format!("{} async {}", vis_snip, &header_snip[vis_snip.len() + 1..ret_pos])
                                };

                                let help = format!("make the function `async` and {ret_sugg}");
                                diag.span_suggestion(
                                    header_span,
                                    help,
                                    format!("{header_snip}{ret_snip}"),
                                    Applicability::MachineApplicable
                                );

                                let body_snip = snippet_block(cx, closure_body.value.span, "..", Some(block.span));
                                diag.span_suggestion(
                                    block.span,
                                    "move the body of the async block to the enclosing function",
                                    body_snip,
                                    Applicability::MachineApplicable
                                );
                            }
                        }
                    },
                );
            }
        }
    }
}

fn future_trait_ref<'tcx>(
    cx: &LateContext<'tcx>,
    ty: &'tcx Ty<'tcx>,
) -> Option<(&'tcx TraitRef<'tcx>, Vec<LifetimeName>)> {
    if_chain! {
        if let TyKind::OpaqueDef(item_id, bounds, false) = ty.kind;
        let item = cx.tcx.hir().item(item_id);
        if let ItemKind::OpaqueTy(opaque) = &item.kind;
        if let Some(trait_ref) = opaque.bounds.iter().find_map(|bound| {
            if let GenericBound::Trait(poly, _) = bound {
                Some(&poly.trait_ref)
            } else {
                None
            }
        });
        if trait_ref.trait_def_id() == cx.tcx.lang_items().future_trait();
        then {
            let output_lifetimes = bounds
                .iter()
                .filter_map(|bound| {
                    if let GenericArg::Lifetime(lt) = bound {
                        Some(lt.res)
                    } else {
                        None
                    }
                })
                .collect();

            return Some((trait_ref, output_lifetimes));
        }
    }

    None
}

fn future_output_ty<'tcx>(trait_ref: &'tcx TraitRef<'tcx>) -> Option<&'tcx Ty<'tcx>> {
    if_chain! {
        if let Some(segment) = trait_ref.path.segments.last();
        if let Some(args) = segment.args;
        if args.bindings.len() == 1;
        let binding = &args.bindings[0];
        if binding.ident.name == sym::Output;
        if let TypeBindingKind::Equality { term: Term::Ty(output) } = binding.kind;
        then {
            return Some(output);
        }
    }

    None
}

fn captures_all_lifetimes(inputs: &[Ty<'_>], output_lifetimes: &[LifetimeName]) -> bool {
    let input_lifetimes: Vec<LifetimeName> = inputs
        .iter()
        .filter_map(|ty| {
            if let TyKind::Ref(lt, _) = ty.kind {
                Some(lt.res)
            } else {
                None
            }
        })
        .collect();

    // The lint should trigger in one of these cases:
    // - There are no input lifetimes
    // - There's only one output lifetime bound using `+ '_`
    // - All input lifetimes are explicitly bound to the output
    input_lifetimes.is_empty()
        || (output_lifetimes.len() == 1 && matches!(output_lifetimes[0], LifetimeName::Infer))
        || input_lifetimes
            .iter()
            .all(|in_lt| output_lifetimes.iter().any(|out_lt| in_lt == out_lt))
}

fn desugared_async_block<'tcx>(cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) -> Option<&'tcx Body<'tcx>> {
    if_chain! {
        if let Some(block_expr) = block.expr;
        if let Expr {
            kind: ExprKind::Closure(&Closure { body, .. }),
            ..
        } = block_expr;
        let closure_body = cx.tcx.hir().body(body);
        if closure_body.generator_kind == Some(GeneratorKind::Async(AsyncGeneratorKind::Block));
        then {
            return Some(closure_body);
        }
    }

    None
}

fn suggested_ret(cx: &LateContext<'_>, output: &Ty<'_>) -> Option<(&'static str, String)> {
    match output.kind {
        TyKind::Tup(tys) if tys.is_empty() => {
            let sugg = "remove the return type";
            Some((sugg, String::new()))
        },
        _ => {
            let sugg = "return the output of the future directly";
            snippet_opt(cx, output.span).map(|snip| (sugg, format!(" -> {snip}")))
        },
    }
}
