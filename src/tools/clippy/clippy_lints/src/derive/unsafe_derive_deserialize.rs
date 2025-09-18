use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::{is_lint_allowed, paths};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{FnKind, Visitor, walk_expr, walk_fn, walk_item};
use rustc_hir::{self as hir, BlockCheckMode, BodyId, Expr, ExprKind, FnDecl, Item, UnsafeSource};
use rustc_lint::LateContext;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, Ty};
use rustc_span::{Span, sym};

use super::UNSAFE_DERIVE_DESERIALIZE;

/// Implementation of the `UNSAFE_DERIVE_DESERIALIZE` lint.
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, item: &Item<'_>, trait_ref: &hir::TraitRef<'_>, ty: Ty<'tcx>) {
    fn has_unsafe<'tcx>(cx: &LateContext<'tcx>, item: &'tcx Item<'_>) -> bool {
        let mut visitor = UnsafeVisitor { cx };
        walk_item(&mut visitor, item).is_break()
    }

    if let Some(trait_def_id) = trait_ref.trait_def_id()
        && paths::SERDE_DESERIALIZE.matches(cx, trait_def_id)
        && let ty::Adt(def, _) = ty.kind()
        && let Some(local_def_id) = def.did().as_local()
        && let adt_hir_id = cx.tcx.local_def_id_to_hir_id(local_def_id)
        && !is_lint_allowed(cx, UNSAFE_DERIVE_DESERIALIZE, adt_hir_id)
        && cx
            .tcx
            .inherent_impls(def.did())
            .iter()
            .map(|imp_did| cx.tcx.hir_expect_item(imp_did.expect_local()))
            .any(|imp| has_unsafe(cx, imp))
    {
        span_lint_hir_and_then(
            cx,
            UNSAFE_DERIVE_DESERIALIZE,
            adt_hir_id,
            item.span,
            "you are deriving `serde::Deserialize` on a type that has methods using `unsafe`",
            |diag| {
                diag.help(
                    "consider implementing `serde::Deserialize` manually. See https://serde.rs/impl-deserialize.html",
                );
            },
        );
    }
}

struct UnsafeVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
}

impl<'tcx> Visitor<'tcx> for UnsafeVisitor<'_, 'tcx> {
    type Result = ControlFlow<()>;
    type NestedFilter = nested_filter::All;

    fn visit_fn(
        &mut self,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body_id: BodyId,
        _: Span,
        id: LocalDefId,
    ) -> Self::Result {
        if let Some(header) = kind.header()
            && header.is_unsafe()
        {
            ControlFlow::Break(())
        } else {
            walk_fn(self, kind, decl, body_id, id)
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) -> Self::Result {
        if let ExprKind::Block(block, _) = expr.kind
            && block.rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)
            && block
                .span
                .source_callee()
                .and_then(|expr| expr.macro_def_id)
                .is_none_or(|did| !self.cx.tcx.is_diagnostic_item(sym::pin_macro, did))
        {
            return ControlFlow::Break(());
        }

        walk_expr(self, expr)
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}
