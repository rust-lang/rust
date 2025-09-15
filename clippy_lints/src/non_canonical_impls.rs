use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::res::MaybeQPath;
use clippy_utils::ty::implements_trait;
use clippy_utils::{is_diag_trait_item, is_from_proc_macro, is_res_lang_ctor, last_path_segment, std_or_core};
use rustc_errors::Applicability;
use rustc_hir::{Block, Body, Expr, ExprKind, ImplItem, ImplItemKind, Item, LangItem, Node, UnOp};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{EarlyBinder, TraitRef, TypeckResults};
use rustc_session::declare_lint_pass;
use rustc_span::sym;
use rustc_span::symbol::kw;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for non-canonical implementations of `Clone` when `Copy` is already implemented.
    ///
    /// ### Why is this bad?
    /// If both `Clone` and `Copy` are implemented, they must agree. This can done by dereferencing
    /// `self` in `Clone`'s implementation, which will avoid any possibility of the implementations
    /// becoming out of sync.
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
    pub NON_CANONICAL_CLONE_IMPL,
    suspicious,
    "non-canonical implementation of `Clone` on a `Copy` type"
}
declare_clippy_lint! {
    /// ### What it does
    /// Checks for non-canonical implementations of `PartialOrd` when `Ord` is already implemented.
    ///
    /// ### Why is this bad?
    /// If both `PartialOrd` and `Ord` are implemented, they must agree. This is commonly done by
    /// wrapping the result of `cmp` in `Some` for `partial_cmp`. Not doing this may silently
    /// introduce an error upon refactoring.
    ///
    /// ### Known issues
    /// Code that calls the `.into()` method instead will be flagged, despite `.into()` wrapping it
    /// in `Some`.
    ///
    /// ### Example
    /// ```no_run
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
    /// ```no_run
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
    ///         Some(self.cmp(other))   // or self.cmp(other).into()
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.73.0"]
    pub NON_CANONICAL_PARTIAL_ORD_IMPL,
    suspicious,
    "non-canonical implementation of `PartialOrd` on an `Ord` type"
}
declare_lint_pass!(NonCanonicalImpls => [NON_CANONICAL_CLONE_IMPL, NON_CANONICAL_PARTIAL_ORD_IMPL]);

impl LateLintPass<'_> for NonCanonicalImpls {
    fn check_impl_item<'tcx>(&mut self, cx: &LateContext<'tcx>, impl_item: &ImplItem<'tcx>) {
        if let ImplItemKind::Fn(_, impl_item_id) = impl_item.kind
            && let Node::Item(item) = cx.tcx.parent_hir_node(impl_item.hir_id())
            && let Some(trait_impl) = cx.tcx.impl_trait_ref(item.owner_id).map(EarlyBinder::skip_binder)
            && let trait_name = cx.tcx.get_diagnostic_name(trait_impl.def_id)
            // NOTE: check this early to avoid expensive checks that come after this one
            && matches!(trait_name, Some(sym::Clone | sym::PartialOrd))
            && !cx.tcx.is_automatically_derived(item.owner_id.to_def_id())
            && let body = cx.tcx.hir_body(impl_item_id)
            && let ExprKind::Block(block, ..) = body.value.kind
            && !block.span.in_external_macro(cx.sess().source_map())
            && !is_from_proc_macro(cx, impl_item)
        {
            if trait_name == Some(sym::Clone)
                && let Some(copy_def_id) = cx.tcx.get_diagnostic_item(sym::Copy)
                && implements_trait(cx, trait_impl.self_ty(), copy_def_id, &[])
            {
                check_clone_on_copy(cx, impl_item, block);
            } else if trait_name == Some(sym::PartialOrd)
                && impl_item.ident.name == sym::partial_cmp
                && let Some(ord_def_id) = cx.tcx.get_diagnostic_item(sym::Ord)
                && implements_trait(cx, trait_impl.self_ty(), ord_def_id, &[])
            {
                check_partial_ord_on_ord(cx, impl_item, item, &trait_impl, body, block);
            }
        }
    }
}

fn check_clone_on_copy(cx: &LateContext<'_>, impl_item: &ImplItem<'_>, block: &Block<'_>) {
    if impl_item.ident.name == sym::clone {
        if block.stmts.is_empty()
            && let Some(expr) = block.expr
            && let ExprKind::Unary(UnOp::Deref, deref) = expr.kind
            && let ExprKind::Path(qpath) = deref.kind
            && last_path_segment(&qpath).ident.name == kw::SelfLower
        {
            // this is the canonical implementation, `fn clone(&self) -> Self { *self }`
            return;
        }

        span_lint_and_sugg(
            cx,
            NON_CANONICAL_CLONE_IMPL,
            block.span,
            "non-canonical implementation of `clone` on a `Copy` type",
            "change this to",
            "{ *self }".to_owned(),
            Applicability::MaybeIncorrect,
        );
    }

    if impl_item.ident.name == sym::clone_from {
        span_lint_and_sugg(
            cx,
            NON_CANONICAL_CLONE_IMPL,
            impl_item.span,
            "unnecessary implementation of `clone_from` on a `Copy` type",
            "remove it",
            String::new(),
            Applicability::MaybeIncorrect,
        );
    }
}

fn check_partial_ord_on_ord<'tcx>(
    cx: &LateContext<'tcx>,
    impl_item: &ImplItem<'_>,
    item: &Item<'_>,
    trait_impl: &TraitRef<'_>,
    body: &Body<'_>,
    block: &Block<'tcx>,
) {
    // If the `cmp` call likely needs to be fully qualified in the suggestion
    // (like `std::cmp::Ord::cmp`). It's unfortunate we must put this here but we can't
    // access `cmp_expr` in the suggestion without major changes, as we lint in `else`.

    let mut needs_fully_qualified = false;
    if block.stmts.is_empty()
        && let Some(expr) = block.expr
        && expr_is_cmp(cx, expr, impl_item, &mut needs_fully_qualified)
    {
        return;
    }
    // Fix #12683, allow [`needless_return`] here
    else if block.expr.is_none()
        && let Some(stmt) = block.stmts.first()
        && let rustc_hir::StmtKind::Semi(Expr {
            kind: ExprKind::Ret(Some(ret)),
            ..
        }) = stmt.kind
        && expr_is_cmp(cx, ret, impl_item, &mut needs_fully_qualified)
    {
        return;
    }
    // If `Self` and `Rhs` are not the same type, bail. This makes creating a valid
    // suggestion tons more complex.
    else if let [lhs, rhs, ..] = trait_impl.args.as_slice()
        && lhs != rhs
    {
        return;
    }

    span_lint_and_then(
        cx,
        NON_CANONICAL_PARTIAL_ORD_IMPL,
        item.span,
        "non-canonical implementation of `partial_cmp` on an `Ord` type",
        |diag| {
            let [_, other] = body.params else {
                return;
            };
            let Some(std_or_core) = std_or_core(cx) else {
                return;
            };

            let suggs = match (other.pat.simple_ident(), needs_fully_qualified) {
                (Some(other_ident), true) => vec![(
                    block.span,
                    format!("{{ Some({std_or_core}::cmp::Ord::cmp(self, {})) }}", other_ident.name),
                )],
                (Some(other_ident), false) => {
                    vec![(block.span, format!("{{ Some(self.cmp({})) }}", other_ident.name))]
                },
                (None, true) => vec![
                    (
                        block.span,
                        format!("{{ Some({std_or_core}::cmp::Ord::cmp(self, other)) }}"),
                    ),
                    (other.pat.span, "other".to_owned()),
                ],
                (None, false) => vec![
                    (block.span, "{ Some(self.cmp(other)) }".to_owned()),
                    (other.pat.span, "other".to_owned()),
                ],
            };

            diag.multipart_suggestion("change this to", suggs, Applicability::Unspecified);
        },
    );
}

/// Return true if `expr_kind` is a `cmp` call.
fn expr_is_cmp<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    impl_item: &ImplItem<'_>,
    needs_fully_qualified: &mut bool,
) -> bool {
    let typeck = cx.tcx.typeck(impl_item.owner_id.def_id);
    match expr.kind {
        ExprKind::Call(
            Expr {
                kind: ExprKind::Path(some_path),
                hir_id: some_hir_id,
                ..
            },
            [cmp_expr],
        ) => {
            is_res_lang_ctor(cx, typeck.qpath_res(some_path, *some_hir_id), LangItem::OptionSome)
                // Fix #11178, allow `Self::cmp(self, ..)`
                && self_cmp_call(cx, typeck, cmp_expr, needs_fully_qualified)
        },
        ExprKind::MethodCall(_, recv, [], _) => {
            typeck
                .type_dependent_def_id(expr.hir_id)
                .is_some_and(|def_id| is_diag_trait_item(cx, def_id, sym::Into))
                && self_cmp_call(cx, typeck, recv, needs_fully_qualified)
        },
        _ => false,
    }
}

/// Returns whether this is any of `self.cmp(..)`, `Self::cmp(self, ..)` or `Ord::cmp(self, ..)`.
fn self_cmp_call<'tcx>(
    cx: &LateContext<'tcx>,
    typeck: &TypeckResults<'tcx>,
    cmp_expr: &'tcx Expr<'tcx>,
    needs_fully_qualified: &mut bool,
) -> bool {
    match cmp_expr.kind {
        ExprKind::Call(path, [_, _]) => path
            .res(typeck)
            .opt_def_id()
            .is_some_and(|def_id| cx.tcx.is_diagnostic_item(sym::ord_cmp_method, def_id)),
        ExprKind::MethodCall(_, recv, [_], ..) => {
            let ExprKind::Path(path) = recv.kind else {
                return false;
            };
            if last_path_segment(&path).ident.name != kw::SelfLower {
                return false;
            }

            // We can set this to true here no matter what as if it's a `MethodCall` and goes to the
            // `else` branch, it must be a method named `cmp` that isn't `Ord::cmp`
            *needs_fully_qualified = true;

            typeck
                .type_dependent_def_id(cmp_expr.hir_id)
                .is_some_and(|def_id| cx.tcx.is_diagnostic_item(sym::ord_cmp_method, def_id))
        },
        _ => false,
    }
}
