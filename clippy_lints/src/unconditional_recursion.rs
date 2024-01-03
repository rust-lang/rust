use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{expr_or_init, get_trait_def_id};
use rustc_ast::BinOpKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{walk_body, FnKind};
use rustc_hir::{Body, Expr, ExprKind, FnDecl, Item, ItemKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::Ident;
use rustc_span::{sym, Span};
use rustc_trait_selection::traits::error_reporting::suggestions::ReturnsVisitor;

declare_clippy_lint! {
    /// ### What it does
    /// Checks that there isn't an infinite recursion in `PartialEq` trait
    /// implementation.
    ///
    /// ### Why is this bad?
    /// This is a hard to find infinite recursion which will crashing any code
    /// using it.
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
    /// Use instead:
    ///
    /// In such cases, either use `#[derive(PartialEq)]` or don't implement it.
    #[clippy::version = "1.76.0"]
    pub UNCONDITIONAL_RECURSION,
    suspicious,
    "detect unconditional recursion in some traits implementation"
}

declare_lint_pass!(UnconditionalRecursion => [UNCONDITIONAL_RECURSION]);

fn get_ty_def_id(ty: Ty<'_>) -> Option<DefId> {
    match ty.peel_refs().kind() {
        ty::Adt(adt, _) => Some(adt.did()),
        ty::Foreign(def_id) => Some(*def_id),
        _ => None,
    }
}

fn has_conditional_return(body: &Body<'_>, expr: &Expr<'_>) -> bool {
    let mut visitor = ReturnsVisitor::default();

    walk_body(&mut visitor, body);
    match visitor.returns.as_slice() {
        [] => false,
        [return_expr] => return_expr.hir_id != expr.hir_id,
        _ => true,
    }
}

#[allow(clippy::unnecessary_def_path)]
fn check_partial_eq(cx: &LateContext<'_>, method_span: Span, method_def_id: LocalDefId, name: Ident, expr: &Expr<'_>) {
    let args = cx
        .tcx
        .instantiate_bound_regions_with_erased(cx.tcx.fn_sig(method_def_id).skip_binder())
        .inputs();
    // That has two arguments.
    if let [self_arg, other_arg] = args
        && let Some(self_arg) = get_ty_def_id(*self_arg)
        && let Some(other_arg) = get_ty_def_id(*other_arg)
        // The two arguments are of the same type.
        && self_arg == other_arg
        && let hir_id = cx.tcx.local_def_id_to_hir_id(method_def_id)
        && let Some((
            _,
            Node::Item(Item {
                kind: ItemKind::Impl(impl_),
                owner_id,
                ..
            }),
        )) = cx.tcx.hir().parent_iter(hir_id).next()
        // We exclude `impl` blocks generated from rustc's proc macros.
        && !cx.tcx.has_attr(*owner_id, sym::automatically_derived)
        // It is a implementation of a trait.
        && let Some(trait_) = impl_.of_trait
        && let Some(trait_def_id) = trait_.trait_def_id()
        // The trait is `PartialEq`.
        && Some(trait_def_id) == get_trait_def_id(cx, &["core", "cmp", "PartialEq"])
    {
        let to_check_op = if name.name == sym::eq {
            BinOpKind::Eq
        } else {
            BinOpKind::Ne
        };
        let is_bad = match expr.kind {
            ExprKind::Binary(op, left, right) if op.node == to_check_op => {
                // Then we check if the left-hand element is of the same type as `self`.
                if let Some(left_ty) = cx.typeck_results().expr_ty_opt(left)
                    && let Some(left_id) = get_ty_def_id(left_ty)
                    && self_arg == left_id
                    && let Some(right_ty) = cx.typeck_results().expr_ty_opt(right)
                    && let Some(right_id) = get_ty_def_id(right_ty)
                    && other_arg == right_id
                {
                    true
                } else {
                    false
                }
            },
            ExprKind::MethodCall(segment, _receiver, &[_arg], _) if segment.ident.name == name.name => {
                if let Some(fn_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
                    && let Some(trait_id) = cx.tcx.trait_of_item(fn_id)
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
    }
}

#[allow(clippy::unnecessary_def_path)]
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
        )) = cx.tcx.hir().parent_iter(hir_id).next()
        // We exclude `impl` blocks generated from rustc's proc macros.
        && !cx.tcx.has_attr(*owner_id, sym::automatically_derived)
        // It is a implementation of a trait.
        && let Some(trait_) = impl_.of_trait
        && let Some(trait_def_id) = trait_.trait_def_id()
        // The trait is `ToString`.
        && Some(trait_def_id) == get_trait_def_id(cx, &["alloc", "string", "ToString"])
    {
        let is_bad = match expr.kind {
            ExprKind::MethodCall(segment, _receiver, &[_arg], _) if segment.ident.name == name.name => {
                if let Some(fn_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
                    && let Some(trait_id) = cx.tcx.trait_of_item(fn_id)
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
    }
}

impl<'tcx> LateLintPass<'tcx> for UnconditionalRecursion {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        _decl: &'tcx FnDecl<'tcx>,
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
            if name.name == sym::eq || name.name == sym::ne {
                check_partial_eq(cx, method_span, method_def_id, name, expr);
            } else if name.name == sym::to_string {
                check_to_string(cx, method_span, method_def_id, name, expr);
            }
        }
    }
}
