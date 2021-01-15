use crate::utils::{get_item_name, snippet_with_applicability, span_lint, span_lint_and_sugg};
use rustc_ast::ast::LitKind;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{AssocItemKind, BinOpKind, Expr, ExprKind, Impl, ImplItemRef, Item, ItemKind, TraitItemRef};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::{Span, Spanned, Symbol};

declare_clippy_lint! {
    /// **What it does:** Checks for getting the length of something via `.len()`
    /// just to compare to zero, and suggests using `.is_empty()` where applicable.
    ///
    /// **Why is this bad?** Some structures can answer `.is_empty()` much faster
    /// than calculating their length. So it is good to get into the habit of using
    /// `.is_empty()`, and having it is cheap.
    /// Besides, it makes the intent clearer than a manual comparison in some contexts.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// if x.len() == 0 {
    ///     ..
    /// }
    /// if y.len() != 0 {
    ///     ..
    /// }
    /// ```
    /// instead use
    /// ```ignore
    /// if x.is_empty() {
    ///     ..
    /// }
    /// if !y.is_empty() {
    ///     ..
    /// }
    /// ```
    pub LEN_ZERO,
    style,
    "checking `.len() == 0` or `.len() > 0` (or similar) when `.is_empty()` could be used instead"
}

declare_clippy_lint! {
    /// **What it does:** Checks for items that implement `.len()` but not
    /// `.is_empty()`.
    ///
    /// **Why is this bad?** It is good custom to have both methods, because for
    /// some data structures, asking about the length will be a costly operation,
    /// whereas `.is_empty()` can usually answer in constant time. Also it used to
    /// lead to false positives on the [`len_zero`](#len_zero) lint – currently that
    /// lint will ignore such entities.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// impl X {
    ///     pub fn len(&self) -> usize {
    ///         ..
    ///     }
    /// }
    /// ```
    pub LEN_WITHOUT_IS_EMPTY,
    style,
    "traits or impls with a public `len` method but no corresponding `is_empty` method"
}

declare_clippy_lint! {
    /// **What it does:** Checks for comparing to an empty slice such as `""` or `[]`,
    /// and suggests using `.is_empty()` where applicable.
    ///
    /// **Why is this bad?** Some structures can answer `.is_empty()` much faster
    /// than checking for equality. So it is good to get into the habit of using
    /// `.is_empty()`, and having it is cheap.
    /// Besides, it makes the intent clearer than a manual comparison in some contexts.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```ignore
    /// if s == "" {
    ///     ..
    /// }
    ///
    /// if arr == [] {
    ///     ..
    /// }
    /// ```
    /// Use instead:
    /// ```ignore
    /// if s.is_empty() {
    ///     ..
    /// }
    ///
    /// if arr.is_empty() {
    ///     ..
    /// }
    /// ```
    pub COMPARISON_TO_EMPTY,
    style,
    "checking `x == \"\"` or `x == []` (or similar) when `.is_empty()` could be used instead"
}

declare_lint_pass!(LenZero => [LEN_ZERO, LEN_WITHOUT_IS_EMPTY, COMPARISON_TO_EMPTY]);

impl<'tcx> LateLintPass<'tcx> for LenZero {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if item.span.from_expansion() {
            return;
        }

        match item.kind {
            ItemKind::Trait(_, _, _, _, ref trait_items) => check_trait_items(cx, item, trait_items),
            ItemKind::Impl(Impl {
                of_trait: None,
                items: ref impl_items,
                ..
            }) => check_impl_items(cx, item, impl_items),
            _ => (),
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        if let ExprKind::Binary(Spanned { node: cmp, .. }, ref left, ref right) = expr.kind {
            match cmp {
                BinOpKind::Eq => {
                    check_cmp(cx, expr.span, left, right, "", 0); // len == 0
                    check_cmp(cx, expr.span, right, left, "", 0); // 0 == len
                },
                BinOpKind::Ne => {
                    check_cmp(cx, expr.span, left, right, "!", 0); // len != 0
                    check_cmp(cx, expr.span, right, left, "!", 0); // 0 != len
                },
                BinOpKind::Gt => {
                    check_cmp(cx, expr.span, left, right, "!", 0); // len > 0
                    check_cmp(cx, expr.span, right, left, "", 1); // 1 > len
                },
                BinOpKind::Lt => {
                    check_cmp(cx, expr.span, left, right, "", 1); // len < 1
                    check_cmp(cx, expr.span, right, left, "!", 0); // 0 < len
                },
                BinOpKind::Ge => check_cmp(cx, expr.span, left, right, "!", 1), // len >= 1
                BinOpKind::Le => check_cmp(cx, expr.span, right, left, "!", 1), // 1 <= len
                _ => (),
            }
        }
    }
}

fn check_trait_items(cx: &LateContext<'_>, visited_trait: &Item<'_>, trait_items: &[TraitItemRef]) {
    fn is_named_self(cx: &LateContext<'_>, item: &TraitItemRef, name: &str) -> bool {
        item.ident.name.as_str() == name
            && if let AssocItemKind::Fn { has_self } = item.kind {
                has_self && {
                    let did = cx.tcx.hir().local_def_id(item.id.hir_id);
                    cx.tcx.fn_sig(did).inputs().skip_binder().len() == 1
                }
            } else {
                false
            }
    }

    // fill the set with current and super traits
    fn fill_trait_set(traitt: DefId, set: &mut FxHashSet<DefId>, cx: &LateContext<'_>) {
        if set.insert(traitt) {
            for supertrait in rustc_trait_selection::traits::supertrait_def_ids(cx.tcx, traitt) {
                fill_trait_set(supertrait, set, cx);
            }
        }
    }

    if cx.access_levels.is_exported(visited_trait.hir_id) && trait_items.iter().any(|i| is_named_self(cx, i, "len")) {
        let mut current_and_super_traits = FxHashSet::default();
        let visited_trait_def_id = cx.tcx.hir().local_def_id(visited_trait.hir_id);
        fill_trait_set(visited_trait_def_id.to_def_id(), &mut current_and_super_traits, cx);

        let is_empty_method_found = current_and_super_traits
            .iter()
            .flat_map(|&i| cx.tcx.associated_items(i).in_definition_order())
            .any(|i| {
                i.kind == ty::AssocKind::Fn
                    && i.fn_has_self_parameter
                    && i.ident.name == sym!(is_empty)
                    && cx.tcx.fn_sig(i.def_id).inputs().skip_binder().len() == 1
            });

        if !is_empty_method_found {
            span_lint(
                cx,
                LEN_WITHOUT_IS_EMPTY,
                visited_trait.span,
                &format!(
                    "trait `{}` has a `len` method but no (possibly inherited) `is_empty` method",
                    visited_trait.ident.name
                ),
            );
        }
    }
}

fn check_impl_items(cx: &LateContext<'_>, item: &Item<'_>, impl_items: &[ImplItemRef<'_>]) {
    fn is_named_self(cx: &LateContext<'_>, item: &ImplItemRef<'_>, name: &str) -> bool {
        item.ident.name.as_str() == name
            && if let AssocItemKind::Fn { has_self } = item.kind {
                has_self && {
                    let did = cx.tcx.hir().local_def_id(item.id.hir_id);
                    cx.tcx.fn_sig(did).inputs().skip_binder().len() == 1
                }
            } else {
                false
            }
    }

    let is_empty = if let Some(is_empty) = impl_items.iter().find(|i| is_named_self(cx, i, "is_empty")) {
        if cx.access_levels.is_exported(is_empty.id.hir_id) {
            return;
        }
        "a private"
    } else {
        "no corresponding"
    };

    if let Some(i) = impl_items.iter().find(|i| is_named_self(cx, i, "len")) {
        if cx.access_levels.is_exported(i.id.hir_id) {
            let def_id = cx.tcx.hir().local_def_id(item.hir_id);
            let ty = cx.tcx.type_of(def_id);

            span_lint(
                cx,
                LEN_WITHOUT_IS_EMPTY,
                item.span,
                &format!(
                    "item `{}` has a public `len` method but {} `is_empty` method",
                    ty, is_empty
                ),
            );
        }
    }
}

fn check_cmp(cx: &LateContext<'_>, span: Span, method: &Expr<'_>, lit: &Expr<'_>, op: &str, compare_to: u32) {
    if let (&ExprKind::MethodCall(ref method_path, _, ref args, _), &ExprKind::Lit(ref lit)) = (&method.kind, &lit.kind)
    {
        // check if we are in an is_empty() method
        if let Some(name) = get_item_name(cx, method) {
            if name.as_str() == "is_empty" {
                return;
            }
        }

        check_len(cx, span, method_path.ident.name, args, &lit.node, op, compare_to)
    } else {
        check_empty_expr(cx, span, method, lit, op)
    }
}

fn check_len(
    cx: &LateContext<'_>,
    span: Span,
    method_name: Symbol,
    args: &[Expr<'_>],
    lit: &LitKind,
    op: &str,
    compare_to: u32,
) {
    if let LitKind::Int(lit, _) = *lit {
        // check if length is compared to the specified number
        if lit != u128::from(compare_to) {
            return;
        }

        if method_name.as_str() == "len" && args.len() == 1 && has_is_empty(cx, &args[0]) {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                LEN_ZERO,
                span,
                &format!("length comparison to {}", if compare_to == 0 { "zero" } else { "one" }),
                &format!("using `{}is_empty` is clearer and more explicit", op),
                format!(
                    "{}{}.is_empty()",
                    op,
                    snippet_with_applicability(cx, args[0].span, "_", &mut applicability)
                ),
                applicability,
            );
        }
    }
}

fn check_empty_expr(cx: &LateContext<'_>, span: Span, lit1: &Expr<'_>, lit2: &Expr<'_>, op: &str) {
    if (is_empty_array(lit2) || is_empty_string(lit2)) && has_is_empty(cx, lit1) {
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            COMPARISON_TO_EMPTY,
            span,
            "comparison to empty slice",
            &format!("using `{}is_empty` is clearer and more explicit", op),
            format!(
                "{}{}.is_empty()",
                op,
                snippet_with_applicability(cx, lit1.span, "_", &mut applicability)
            ),
            applicability,
        );
    }
}

fn is_empty_string(expr: &Expr<'_>) -> bool {
    if let ExprKind::Lit(ref lit) = expr.kind {
        if let LitKind::Str(lit, _) = lit.node {
            let lit = lit.as_str();
            return lit == "";
        }
    }
    false
}

fn is_empty_array(expr: &Expr<'_>) -> bool {
    if let ExprKind::Array(ref arr) = expr.kind {
        return arr.is_empty();
    }
    false
}

/// Checks if this type has an `is_empty` method.
fn has_is_empty(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    /// Gets an `AssocItem` and return true if it matches `is_empty(self)`.
    fn is_is_empty(cx: &LateContext<'_>, item: &ty::AssocItem) -> bool {
        if let ty::AssocKind::Fn = item.kind {
            if item.ident.name.as_str() == "is_empty" {
                let sig = cx.tcx.fn_sig(item.def_id);
                let ty = sig.skip_binder();
                ty.inputs().len() == 1
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Checks the inherent impl's items for an `is_empty(self)` method.
    fn has_is_empty_impl(cx: &LateContext<'_>, id: DefId) -> bool {
        cx.tcx.inherent_impls(id).iter().any(|imp| {
            cx.tcx
                .associated_items(*imp)
                .in_definition_order()
                .any(|item| is_is_empty(cx, &item))
        })
    }

    let ty = &cx.typeck_results().expr_ty(expr).peel_refs();
    match ty.kind() {
        ty::Dynamic(ref tt, ..) => tt.principal().map_or(false, |principal| {
            cx.tcx
                .associated_items(principal.def_id())
                .in_definition_order()
                .any(|item| is_is_empty(cx, &item))
        }),
        ty::Projection(ref proj) => has_is_empty_impl(cx, proj.item_def_id),
        ty::Adt(id, _) => has_is_empty_impl(cx, id.did),
        ty::Array(..) | ty::Slice(..) | ty::Str => true,
        _ => false,
    }
}
