use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::ty::implements_trait;
use clippy_utils::{get_parent_node, is_res_lang_ctor, last_path_segment, path_res};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{Expr, ExprKind, ImplItem, ImplItemKind, ItemKind, LangItem, Node, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::EarlyBinder;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;
use rustc_span::symbol::kw;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual implementations of `Clone` when `Copy` is already implemented.
    ///
    /// ### Why is this bad?
    /// If both `Clone` and `Copy` are implemented, they must agree. This is done by dereferencing
    /// `self` in `Clone`'s implementation. Anything else is incorrect.
    ///
    /// ### Example
    /// ```rust,ignore
    /// #[derive(Eq, PartialEq)]
    /// struct A(u32);
    ///
    /// impl Clone for A {
    ///     fn clone(&self) -> Self {
    ///         Self(self.0)
    ///     }
    /// }
    ///
    /// impl Copy for A {}
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// #[derive(Eq, PartialEq)]
    /// struct A(u32);
    ///
    /// impl Clone for A {
    ///     fn clone(&self) -> Self {
    ///         *self
    ///     }
    /// }
    ///
    /// impl Copy for A {}
    /// ```
    #[clippy::version = "1.72.0"]
    pub INCORRECT_CLONE_IMPL_ON_COPY_TYPE,
    correctness,
    "manual implementation of `Clone` on a `Copy` type"
}
declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual implementations of both `PartialOrd` and `Ord` when only `Ord` is
    /// necessary.
    ///
    /// ### Why is this bad?
    /// If both `PartialOrd` and `Ord` are implemented, they must agree. This is commonly done by
    /// wrapping the result of `cmp` in `Some` for `partial_cmp`. Not doing this may silently
    /// introduce an error upon refactoring.
    ///
    /// ### Limitations
    /// Will not lint if `Self` and `Rhs` do not have the same type.
    ///
    /// ### Example
    /// ```rust
    /// # use std::cmp::Ordering;
    /// #[derive(Eq, PartialEq)]
    /// struct A(u32);
    ///
    /// impl Ord for A {
    ///     fn cmp(&self, other: &Self) -> Ordering {
    ///         // ...
    /// #       todo!();
    ///     }
    /// }
    ///
    /// impl PartialOrd for A {
    ///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    ///         // ...
    /// #       todo!();
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::cmp::Ordering;
    /// #[derive(Eq, PartialEq)]
    /// struct A(u32);
    ///
    /// impl Ord for A {
    ///     fn cmp(&self, other: &Self) -> Ordering {
    ///         // ...
    /// #       todo!();
    ///     }
    /// }
    ///
    /// impl PartialOrd for A {
    ///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    ///         Some(self.cmp(other))
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.72.0"]
    pub INCORRECT_PARTIAL_ORD_IMPL_ON_ORD_TYPE,
    correctness,
    "manual implementation of `PartialOrd` when `Ord` is already implemented"
}
declare_lint_pass!(IncorrectImpls => [INCORRECT_CLONE_IMPL_ON_COPY_TYPE, INCORRECT_PARTIAL_ORD_IMPL_ON_ORD_TYPE]);

impl LateLintPass<'_> for IncorrectImpls {
    #[expect(clippy::too_many_lines)]
    fn check_impl_item(&mut self, cx: &LateContext<'_>, impl_item: &ImplItem<'_>) {
        let Some(Node::Item(item)) = get_parent_node(cx.tcx, impl_item.hir_id()) else {
            return;
        };
        let Some(trait_impl) = cx.tcx.impl_trait_ref(item.owner_id).map(EarlyBinder::skip_binder) else {
            return;
        };
        if cx.tcx.is_automatically_derived(item.owner_id.to_def_id()) {
            return;
        }
        let ItemKind::Impl(_) = item.kind else {
            return;
        };
        let ImplItemKind::Fn(_, impl_item_id) = cx.tcx.hir().impl_item(impl_item.impl_item_id()).kind else {
            return;
        };
        let body = cx.tcx.hir().body(impl_item_id);
        let ExprKind::Block(block, ..) = body.value.kind else {
            return;
        };

        if cx.tcx.is_diagnostic_item(sym::Clone, trait_impl.def_id)
            && let Some(copy_def_id) = cx.tcx.get_diagnostic_item(sym::Copy)
            && implements_trait(
                    cx,
                    trait_impl.self_ty(),
                    copy_def_id,
                    &[],
                )
        {
            if impl_item.ident.name == sym::clone {
                if block.stmts.is_empty()
                    && let Some(expr) = block.expr
                    && let ExprKind::Unary(UnOp::Deref, deref) = expr.kind
                    && let ExprKind::Path(qpath) = deref.kind
                    && last_path_segment(&qpath).ident.name == kw::SelfLower
                {} else {
                    span_lint_and_sugg(
                        cx,
                        INCORRECT_CLONE_IMPL_ON_COPY_TYPE,
                        block.span,
                        "incorrect implementation of `clone` on a `Copy` type",
                        "change this to",
                        "{ *self }".to_owned(),
                        Applicability::MaybeIncorrect,
                    );

                    return;
                }
            }

            if impl_item.ident.name == sym::clone_from {
                span_lint_and_sugg(
                    cx,
                    INCORRECT_CLONE_IMPL_ON_COPY_TYPE,
                    impl_item.span,
                    "incorrect implementation of `clone_from` on a `Copy` type",
                    "remove it",
                    String::new(),
                    Applicability::MaybeIncorrect,
                );

                return;
            }
        }

        if cx.tcx.is_diagnostic_item(sym::PartialOrd, trait_impl.def_id)
            && impl_item.ident.name == sym::partial_cmp
            && let Some(ord_def_id) = cx
                .tcx
                .diagnostic_items(trait_impl.def_id.krate)
                .name_to_id
                .get(&sym::Ord)
            && implements_trait(
                    cx,
                    trait_impl.self_ty(),
                    *ord_def_id,
                    &[],
                )
        {
            if block.stmts.is_empty()
                && let Some(expr) = block.expr
                && let ExprKind::Call(
                        Expr {
                            kind: ExprKind::Path(some_path),
                            hir_id: some_hir_id,
                            ..
                        },
                        [cmp_expr],
                    ) = expr.kind
                && is_res_lang_ctor(cx, cx.qpath_res(some_path, *some_hir_id), LangItem::OptionSome)
                && let ExprKind::MethodCall(cmp_path, _, [other_expr], ..) = cmp_expr.kind
                && cmp_path.ident.name == sym::cmp
                && let Res::Local(..) = path_res(cx, other_expr)
            {} else {
                // If `Self` and `Rhs` are not the same type, bail. This makes creating a valid
                // suggestion tons more complex.
                if let [lhs, rhs, ..] = trait_impl.args.as_slice() && lhs != rhs {
                    return;
                }

                span_lint_and_then(
                    cx,
                    INCORRECT_PARTIAL_ORD_IMPL_ON_ORD_TYPE,
                    item.span,
                    "incorrect implementation of `partial_cmp` on an `Ord` type",
                    |diag| {
                        let [_, other] = body.params else {
                            return;
                        };

                        let suggs = if let Some(other_ident) = other.pat.simple_ident() {
                            vec![(block.span, format!("{{ Some(self.cmp({})) }}", other_ident.name))]
                        } else {
                            vec![
                                (block.span, "{ Some(self.cmp(other)) }".to_owned()),
                                (other.pat.span, "other".to_owned()),
                            ]
                        };

                        diag.multipart_suggestion(
                            "change this to",
                            suggs,
                            Applicability::Unspecified,
                        );
                    }
                );
            }
        }
    }
}
