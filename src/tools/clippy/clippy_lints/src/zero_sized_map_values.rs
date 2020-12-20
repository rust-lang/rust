use if_chain::if_chain;
use rustc_hir::{self as hir, HirId, ItemKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{Adt, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_target::abi::LayoutOf as _;
use rustc_typeck::hir_ty_to_ty;

use crate::utils::{is_type_diagnostic_item, match_type, paths, span_lint_and_help};

declare_clippy_lint! {
    /// **What it does:** Checks for maps with zero-sized value types anywhere in the code.
    ///
    /// **Why is this bad?** Since there is only a single value for a zero-sized type, a map
    /// containing zero sized values is effectively a set. Using a set in that case improves
    /// readability and communicates intent more clearly.
    ///
    /// **Known problems:**
    /// * A zero-sized type cannot be recovered later if it contains private fields.
    /// * This lints the signature of public items
    ///
    /// **Example:**
    ///
    /// ```rust
    /// # use std::collections::HashMap;
    /// fn unique_words(text: &str) -> HashMap<&str, ()> {
    ///     todo!();
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::collections::HashSet;
    /// fn unique_words(text: &str) -> HashSet<&str> {
    ///     todo!();
    /// }
    /// ```
    pub ZERO_SIZED_MAP_VALUES,
    pedantic,
    "usage of map with zero-sized value type"
}

declare_lint_pass!(ZeroSizedMapValues => [ZERO_SIZED_MAP_VALUES]);

impl LateLintPass<'_> for ZeroSizedMapValues {
    fn check_ty(&mut self, cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>) {
        if_chain! {
            if !hir_ty.span.from_expansion();
            if !in_trait_impl(cx, hir_ty.hir_id);
            let ty = ty_from_hir_ty(cx, hir_ty);
            if is_type_diagnostic_item(cx, ty, sym!(hashmap_type)) || match_type(cx, ty, &paths::BTREEMAP);
            if let Adt(_, ref substs) = ty.kind();
            let ty = substs.type_at(1);
            if let Ok(layout) = cx.layout_of(ty);
            if layout.is_zst();
            then {
                span_lint_and_help(cx, ZERO_SIZED_MAP_VALUES, hir_ty.span, "map with zero-sized value type", None, "consider using a set instead");
            }
        }
    }
}

fn in_trait_impl(cx: &LateContext<'_>, hir_id: HirId) -> bool {
    let parent_id = cx.tcx.hir().get_parent_item(hir_id);
    if let Some(Node::Item(item)) = cx.tcx.hir().find(cx.tcx.hir().get_parent_item(parent_id)) {
        if let ItemKind::Impl { of_trait: Some(_), .. } = item.kind {
            return true;
        }
    }
    false
}

fn ty_from_hir_ty<'tcx>(cx: &LateContext<'tcx>, hir_ty: &hir::Ty<'_>) -> Ty<'tcx> {
    cx.maybe_typeck_results()
        .and_then(|results| {
            if results.hir_owner == hir_ty.hir_id.owner {
                results.node_type_opt(hir_ty.hir_id)
            } else {
                None
            }
        })
        .unwrap_or_else(|| hir_ty_to_ty(cx.tcx, hir_ty))
}
