use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{expr_or_init, fn_def_id_with_node_args, path_def_id};
use rustc_ast::BinOpKind;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{FnKind, Visitor, walk_body, walk_expr};
use rustc_hir::{Body, Expr, ExprKind, FnDecl, HirId, Item, ItemKind, Node, QPath, TyKind};
use rustc_hir_analysis::lower_ty;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::impl_lint_pass;
use rustc_span::symbol::{Ident, kw};
use rustc_span::{Span, sym};
use rustc_trait_selection::error_reporting::traits::suggestions::ReturnsVisitor;
use std::ops::ControlFlow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks that there isn't an infinite recursion in trait
    /// implementations.
    ///
    /// ### Why is this bad?
    /// Infinite recursion in trait implementation will either cause crashes
    /// or result in an infinite loop, and it is hard to detect.
    ///
    /// ### Example
    /// ```no_run
    /// enum Foo {
    ///     A,
    ///     B,
    /// }
    ///
    /// impl PartialEq for Foo {
    ///     fn eq(&self, other: &Self) -> bool {
    ///         self == other // bad!
    ///     }
    /// }
    /// ```
    ///
    /// Use instead:
    ///
    /// ```no_run
    /// #[derive(PartialEq)]
    /// enum Foo {
    ///     A,
    ///     B,
    /// }
    /// ```
    ///
    /// As an alternative, rewrite the logic without recursion:
    ///
    /// ```no_run
    /// enum Foo {
    ///     A,
    ///     B,
    /// }
    ///
    /// impl PartialEq for Foo {
    ///     fn eq(&self, other: &Self) -> bool {
    ///         matches!((self, other), (Foo::A, Foo::A) | (Foo::B, Foo::B))
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.77.0"]
    pub UNCONDITIONAL_RECURSION,
    suspicious,
    "detect unconditional recursion in some traits implementation"
}

#[derive(Default)]
pub struct UnconditionalRecursion {
    /// The key is the `DefId` of the type implementing the `Default` trait and the value is the
    /// `DefId` of the return call.
    default_impl_for_type: FxHashMap<DefId, DefId>,
}

impl_lint_pass!(UnconditionalRecursion => [UNCONDITIONAL_RECURSION]);

fn span_error(cx: &LateContext<'_>, method_span: Span, expr: &Expr<'_>) {
    span_lint_and_then(
        cx,
        UNCONDITIONAL_RECURSION,
        method_span,
        "function cannot return without recursing",
        |diag| {
            diag.span_note(expr.span, "recursive call site");
        },
    );
}

fn get_hir_ty_def_id<'tcx>(tcx: TyCtxt<'tcx>, hir_ty: rustc_hir::Ty<'tcx>) -> Option<DefId> {
    let TyKind::Path(qpath) = hir_ty.kind else { return None };
    match qpath {
        QPath::Resolved(_, path) => path.res.opt_def_id(),
        QPath::TypeRelative(_, _) => {
            let ty = lower_ty(tcx, &hir_ty);

            match ty.kind() {
                ty::Alias(ty::Projection, proj) => {
                    Res::<HirId>::Def(DefKind::Trait, proj.trait_ref(tcx).def_id).opt_def_id()
                },
                _ => None,
            }
        },
        QPath::LangItem(..) => None,
    }
}

fn get_return_calls_in_body<'tcx>(body: &'tcx Body<'tcx>) -> Vec<&'tcx Expr<'tcx>> {
    let mut visitor = ReturnsVisitor::default();

    visitor.visit_body(body);
    visitor.returns
}

fn has_conditional_return(body: &Body<'_>, expr: &Expr<'_>) -> bool {
    match get_return_calls_in_body(body).as_slice() {
        [] => false,
        [return_expr] => return_expr.hir_id != expr.hir_id,
        _ => true,
    }
}

fn get_impl_trait_def_id(cx: &LateContext<'_>, method_def_id: LocalDefId) -> Option<DefId> {
    let hir_id = cx.tcx.local_def_id_to_hir_id(method_def_id);
    if let Some((
        _,
        Node::Item(Item {
            kind: ItemKind::Impl(impl_),
            owner_id,
            ..
        }),
    )) = cx.tcx.hir_parent_iter(hir_id).next()
        // We exclude `impl` blocks generated from rustc's proc macros.
        && !cx.tcx.is_automatically_derived(owner_id.to_def_id())
        // It is a implementation of a trait.
        && let Some(of_trait) = impl_.of_trait
    {
        of_trait.trait_ref.trait_def_id()
    } else {
        None
    }
}

/// When we have `x == y` where `x = &T` and `y = &T`, then that resolves to
/// `<&T as PartialEq<&T>>::eq`, which is not the same as `<T as PartialEq<T>>::eq`,
/// however we still would want to treat it the same, because we know that it's a blanket impl
/// that simply delegates to the `PartialEq` impl with one reference removed.
///
/// Still, we can't just do `lty.peel_refs() == rty.peel_refs()` because when we have `x = &T` and
/// `y = &&T`, this is not necessarily the same as `<T as PartialEq<T>>::eq`
///
/// So to avoid these FNs and FPs, we keep removing a layer of references from *both* sides
/// until both sides match the expected LHS and RHS type (or they don't).
fn matches_ty<'tcx>(
    mut left: Ty<'tcx>,
    mut right: Ty<'tcx>,
    expected_left: Ty<'tcx>,
    expected_right: Ty<'tcx>,
) -> bool {
    while let (&ty::Ref(_, lty, _), &ty::Ref(_, rty, _)) = (left.kind(), right.kind()) {
        if lty == expected_left && rty == expected_right {
            return true;
        }
        left = lty;
        right = rty;
    }
    false
}

fn check_partial_eq(cx: &LateContext<'_>, method_span: Span, method_def_id: LocalDefId, name: Ident, expr: &Expr<'_>) {
    let Some(sig) = cx
        .typeck_results()
        .liberated_fn_sigs()
        .get(cx.tcx.local_def_id_to_hir_id(method_def_id))
    else {
        return;
    };

    // That has two arguments.
    if let [self_arg, other_arg] = sig.inputs()
        && let &ty::Ref(_, self_arg, _) = self_arg.kind()
        && let &ty::Ref(_, other_arg, _) = other_arg.kind()
        // The two arguments are of the same type.
        && let Some(trait_def_id) = get_impl_trait_def_id(cx, method_def_id)
        // The trait is `PartialEq`.
        && cx.tcx.is_diagnostic_item(sym::PartialEq, trait_def_id)
    {
        let to_check_op = if name.name == sym::eq {
            BinOpKind::Eq
        } else {
            BinOpKind::Ne
        };
        let is_bad = match expr.kind {
            ExprKind::Binary(op, left, right) if op.node == to_check_op => {
                // Then we check if the LHS matches self_arg and RHS matches other_arg
                let left_ty = cx.typeck_results().expr_ty_adjusted(left);
                let right_ty = cx.typeck_results().expr_ty_adjusted(right);
                matches_ty(left_ty, right_ty, self_arg, other_arg)
            },
            ExprKind::MethodCall(segment, receiver, [arg], _) if segment.ident.name == name.name => {
                let receiver_ty = cx.typeck_results().expr_ty_adjusted(receiver);
                let arg_ty = cx.typeck_results().expr_ty_adjusted(arg);

                if let Some(fn_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
                    && let Some(trait_id) = cx.tcx.trait_of_assoc(fn_id)
                    && trait_id == trait_def_id
                    && matches_ty(receiver_ty, arg_ty, self_arg, other_arg)
                {
                    true
                } else {
                    false
                }
            },
            _ => false,
        };
        if is_bad {
            span_error(cx, method_span, expr);
        }
    }
}

fn check_to_string(cx: &LateContext<'_>, method_span: Span, method_def_id: LocalDefId, name: Ident, expr: &Expr<'_>) {
    let args = cx
        .tcx
        .instantiate_bound_regions_with_erased(cx.tcx.fn_sig(method_def_id).skip_binder())
        .inputs();
    // That has one argument.
    if let [_self_arg] = args
        && let hir_id = cx.tcx.local_def_id_to_hir_id(method_def_id)
        && let Some((
            _,
            Node::Item(Item {
                kind: ItemKind::Impl(impl_),
                owner_id,
                ..
            }),
        )) = cx.tcx.hir_parent_iter(hir_id).next()
        // We exclude `impl` blocks generated from rustc's proc macros.
        && !cx.tcx.is_automatically_derived(owner_id.to_def_id())
        // It is a implementation of a trait.
        && let Some(of_trait) = impl_.of_trait
        && let Some(trait_def_id) = of_trait.trait_ref.trait_def_id()
        // The trait is `ToString`.
        && cx.tcx.is_diagnostic_item(sym::ToString, trait_def_id)
    {
        let is_bad = match expr.kind {
            ExprKind::MethodCall(segment, _receiver, &[_arg], _) if segment.ident.name == name.name => {
                if let Some(fn_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
                    && let Some(trait_id) = cx.tcx.trait_of_assoc(fn_id)
                    && trait_id == trait_def_id
                {
                    true
                } else {
                    false
                }
            },
            _ => false,
        };
        if is_bad {
            span_error(cx, method_span, expr);
        }
    }
}

fn is_default_method_on_current_ty<'tcx>(tcx: TyCtxt<'tcx>, qpath: QPath<'tcx>, implemented_ty_id: DefId) -> bool {
    match qpath {
        QPath::Resolved(_, path) => match path.segments {
            [first, .., last] => last.ident.name == kw::Default && first.res.opt_def_id() == Some(implemented_ty_id),
            _ => false,
        },
        QPath::TypeRelative(ty, segment) => {
            if segment.ident.name != kw::Default {
                return false;
            }
            if matches!(
                ty.kind,
                TyKind::Path(QPath::Resolved(
                    _,
                    hir::Path {
                        res: Res::SelfTyAlias { .. },
                        ..
                    },
                ))
            ) {
                return true;
            }
            get_hir_ty_def_id(tcx, *ty) == Some(implemented_ty_id)
        },
        QPath::LangItem(..) => false,
    }
}

struct CheckCalls<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    implemented_ty_id: DefId,
    method_span: Span,
}

impl<'a, 'tcx> Visitor<'tcx> for CheckCalls<'a, 'tcx>
where
    'tcx: 'a,
{
    type NestedFilter = nested_filter::OnlyBodies;
    type Result = ControlFlow<()>;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) -> ControlFlow<()> {
        walk_expr(self, expr)?;

        if let ExprKind::Call(f, _) = expr.kind
            && let ExprKind::Path(qpath) = f.kind
            && is_default_method_on_current_ty(self.cx.tcx, qpath, self.implemented_ty_id)
            && let Some(method_def_id) = path_def_id(self.cx, f)
            && let Some(trait_def_id) = self.cx.tcx.trait_of_assoc(method_def_id)
            && self.cx.tcx.is_diagnostic_item(sym::Default, trait_def_id)
        {
            span_error(self.cx, self.method_span, expr);
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

impl UnconditionalRecursion {
    fn init_default_impl_for_type_if_needed(&mut self, cx: &LateContext<'_>) {
        if self.default_impl_for_type.is_empty()
            && let Some(default_trait_id) = cx.tcx.get_diagnostic_item(sym::Default)
        {
            let impls = cx.tcx.trait_impls_of(default_trait_id);
            for (ty, impl_def_ids) in impls.non_blanket_impls() {
                let Some(self_def_id) = ty.def() else { continue };
                for impl_def_id in impl_def_ids {
                    if !cx.tcx.is_automatically_derived(*impl_def_id) &&
                        let Some(assoc_item) = cx
                            .tcx
                            .associated_items(impl_def_id)
                            .in_definition_order()
                            // We're not interested in foreign implementations of the `Default` trait.
                            .find(|item| {
                                item.is_fn() && item.def_id.is_local() && item.name() == kw::Default
                            })
                        && let Some(body_node) = cx.tcx.hir_get_if_local(assoc_item.def_id)
                        && let Some(body_id) = body_node.body_id()
                        && let body = cx.tcx.hir_body(body_id)
                        // We don't want to keep it if it has conditional return.
                        && let [return_expr] = get_return_calls_in_body(body).as_slice()
                        && let ExprKind::Call(call_expr, _) = return_expr.kind
                        // We need to use typeck here to infer the actual function being called.
                        && let body_def_id = cx.tcx.hir_enclosing_body_owner(call_expr.hir_id)
                        && let Some(body_owner) = cx.tcx.hir_maybe_body_owned_by(body_def_id)
                        && let typeck = cx.tcx.typeck_body(body_owner.id())
                        && let Some(call_def_id) = typeck.type_dependent_def_id(call_expr.hir_id)
                    {
                        self.default_impl_for_type.insert(self_def_id, call_def_id);
                    }
                }
            }
        }
    }

    fn check_default_new<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        decl: &FnDecl<'tcx>,
        body: &'tcx Body<'tcx>,
        method_span: Span,
        method_def_id: LocalDefId,
    ) {
        // We're only interested into static methods.
        if decl.implicit_self.has_implicit_self() {
            return;
        }
        // We don't check trait implementations.
        if get_impl_trait_def_id(cx, method_def_id).is_some() {
            return;
        }

        let hir_id = cx.tcx.local_def_id_to_hir_id(method_def_id);
        if let Some((
            _,
            Node::Item(Item {
                kind: ItemKind::Impl(impl_),
                ..
            }),
        )) = cx.tcx.hir_parent_iter(hir_id).next()
            && let Some(implemented_ty_id) = get_hir_ty_def_id(cx.tcx, *impl_.self_ty)
            && {
                self.init_default_impl_for_type_if_needed(cx);
                true
            }
            && let Some(return_def_id) = self.default_impl_for_type.get(&implemented_ty_id)
            && method_def_id.to_def_id() == *return_def_id
        {
            let mut c = CheckCalls {
                cx,
                implemented_ty_id,
                method_span,
            };
            let _ = walk_body(&mut c, body);
        }
    }
}

fn check_from(cx: &LateContext<'_>, method_span: Span, method_def_id: LocalDefId, expr: &Expr<'_>) {
    let Some(sig) = cx
        .typeck_results()
        .liberated_fn_sigs()
        .get(cx.tcx.local_def_id_to_hir_id(method_def_id))
    else {
        return;
    };

    // Check if we are calling `Into::into` where the node args match with our `From::from` signature:
    // From::from signature: fn(S1) -> S2
    // <S1 as Into<S2>>::into(s1), node_args=[S1, S2]
    // If they do match, then it must mean that it is the blanket impl,
    // which calls back into our `From::from` again (`Into` is not specializable).
    // rustc's unconditional_recursion already catches calling `From::from` directly
    if let Some((fn_def_id, node_args)) = fn_def_id_with_node_args(cx, expr)
        && let [s1, s2] = **node_args
        && let (Some(s1), Some(s2)) = (s1.as_type(), s2.as_type())
        && let Some(trait_def_id) = cx.tcx.trait_of_assoc(fn_def_id)
        && cx.tcx.is_diagnostic_item(sym::Into, trait_def_id)
        && get_impl_trait_def_id(cx, method_def_id) == cx.tcx.get_diagnostic_item(sym::From)
        && s1 == sig.inputs()[0]
        && s2 == sig.output()
    {
        span_error(cx, method_span, expr);
    }
}

impl<'tcx> LateLintPass<'tcx> for UnconditionalRecursion {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'tcx>,
        body: &'tcx Body<'tcx>,
        method_span: Span,
        method_def_id: LocalDefId,
    ) {
        // If the function is a method...
        if let FnKind::Method(name, _) = kind
            && let expr = expr_or_init(cx, body.value).peel_blocks()
            // Doesn't have a conditional return.
            && !has_conditional_return(body, expr)
        {
            match name.name {
                sym::eq | sym::ne => check_partial_eq(cx, method_span, method_def_id, name, expr),
                sym::to_string => check_to_string(cx, method_span, method_def_id, name, expr),
                sym::from => check_from(cx, method_span, method_def_id, expr),
                _ => {},
            }
            self.check_default_new(cx, decl, body, method_span, method_def_id);
        }
    }
}
