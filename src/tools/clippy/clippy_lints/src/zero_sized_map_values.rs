use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::{is_normalizable, is_type_diagnostic_item};
use rustc_hir::{self as hir, HirId, ItemKind, Node};
use rustc_hir_analysis::lower_ty;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::layout::LayoutOf as _;
use rustc_middle::ty::{self, Ty, TypeVisitableExt};
use rustc_session::declare_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for maps with zero-sized value types anywhere in the code.
    ///
    /// ### Why is this bad?
    /// Since there is only a single value for a zero-sized type, a map
    /// containing zero sized values is effectively a set. Using a set in that case improves
    /// readability and communicates intent more clearly.
    ///
    /// ### Known problems
    /// * A zero-sized type cannot be recovered later if it contains private fields.
    /// * This lints the signature of public items
    ///
    /// ### Example
    /// ```no_run
    /// # use std::collections::HashMap;
    /// fn unique_words(text: &str) -> HashMap<&str, ()> {
    ///     todo!();
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::collections::HashSet;
    /// fn unique_words(text: &str) -> HashSet<&str> {
    ///     todo!();
    /// }
    /// ```
    #[clippy::version = "1.50.0"]
    pub ZERO_SIZED_MAP_VALUES,
    pedantic,
    "usage of map with zero-sized value type"
}

declare_lint_pass!(ZeroSizedMapValues => [ZERO_SIZED_MAP_VALUES]);

impl LateLintPass<'_> for ZeroSizedMapValues {
    fn check_ty<'tcx>(&mut self, cx: &LateContext<'tcx>, hir_ty: &hir::Ty<'tcx>) {
        if !hir_ty.span.from_expansion()
            && !in_trait_impl(cx, hir_ty.hir_id)
            && let ty = ty_from_hir_ty(cx, hir_ty)
            && (is_type_diagnostic_item(cx, ty, sym::HashMap) || is_type_diagnostic_item(cx, ty, sym::BTreeMap))
            && let ty::Adt(_, args) = ty.kind()
            && let ty = args.type_at(1)
            // Fixes https://github.com/rust-lang/rust-clippy/issues/7447 because of
            // https://github.com/rust-lang/rust/blob/master/compiler/rustc_middle/src/ty/sty.rs#L968
            && !ty.has_escaping_bound_vars()
            // Do this to prevent `layout_of` crashing, being unable to fully normalize `ty`.
            && is_normalizable(cx, cx.param_env, ty)
            && let Ok(layout) = cx.layout_of(ty)
            && layout.is_zst()
        {
            span_lint_and_help(
                cx,
                ZERO_SIZED_MAP_VALUES,
                hir_ty.span,
                "map with zero-sized value type",
                None,
                "consider using a set instead",
            );
        }
    }
}

fn in_trait_impl(cx: &LateContext<'_>, hir_id: HirId) -> bool {
    let parent_id = cx.tcx.hir().get_parent_item(hir_id);
    let second_parent_id = cx.tcx.hir().get_parent_item(parent_id.into()).def_id;
    if let Node::Item(item) = cx.tcx.hir_node_by_def_id(second_parent_id) {
        if let ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }) = item.kind {
            return true;
        }
    }
    false
}

fn ty_from_hir_ty<'tcx>(cx: &LateContext<'tcx>, hir_ty: &hir::Ty<'tcx>) -> Ty<'tcx> {
    cx.maybe_typeck_results()
        .and_then(|results| {
            if results.hir_owner == hir_ty.hir_id.owner {
                results.node_type_opt(hir_ty.hir_id)
            } else {
                None
            }
        })
        .unwrap_or_else(|| lower_ty(cx.tcx, hir_ty))
}
