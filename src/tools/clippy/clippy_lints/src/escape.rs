use clippy_utils::diagnostics::span_lint_hir;
use rustc_hir::intravisit;
use rustc_hir::{self, AssocItemKind, Body, FnDecl, HirId, HirIdSet, Impl, ItemKind, Node, Pat, PatKind};
use rustc_hir_typeck::expr_use_visitor::{Delegate, ExprUseVisitor, PlaceBase, PlaceWithHirId};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, TraitRef, Ty};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Span;
use rustc_span::symbol::kw;
use rustc_target::spec::abi::Abi;

#[derive(Copy, Clone)]
pub struct BoxedLocal {
    pub too_large_for_stack: u64,
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `Box<T>` where an unboxed `T` would
    /// work fine.
    ///
    /// ### Why is this bad?
    /// This is an unnecessary allocation, and bad for
    /// performance. It is only necessary to allocate if you wish to move the box
    /// into something.
    ///
    /// ### Example
    /// ```rust
    /// fn foo(x: Box<u32>) {}
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// fn foo(x: u32) {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub BOXED_LOCAL,
    perf,
    "using `Box<T>` where unnecessary"
}

fn is_non_trait_box(ty: Ty<'_>) -> bool {
    ty.is_box() && !ty.boxed_ty().is_trait()
}

struct EscapeDelegate<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    set: HirIdSet,
    trait_self_ty: Option<Ty<'tcx>>,
    too_large_for_stack: u64,
}

impl_lint_pass!(BoxedLocal => [BOXED_LOCAL]);

impl<'tcx> LateLintPass<'tcx> for BoxedLocal {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: intravisit::FnKind<'tcx>,
        _: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        _: Span,
        hir_id: HirId,
    ) {
        if let Some(header) = fn_kind.header() {
            if header.abi != Abi::Rust {
                return;
            }
        }

        let parent_id = cx.tcx.hir().get_parent_item(hir_id).def_id;
        let parent_node = cx.tcx.hir().find_by_def_id(parent_id);

        let mut trait_self_ty = None;
        if let Some(Node::Item(item)) = parent_node {
            // If the method is an impl for a trait, don't warn.
            if let ItemKind::Impl(Impl { of_trait: Some(_), .. }) = item.kind {
                return;
            }

            // find `self` ty for this trait if relevant
            if let ItemKind::Trait(_, _, _, _, items) = item.kind {
                for trait_item in items {
                    if trait_item.id.hir_id() == hir_id {
                        // be sure we have `self` parameter in this function
                        if trait_item.kind == (AssocItemKind::Fn { has_self: true }) {
                            trait_self_ty = Some(
                                TraitRef::identity(cx.tcx, trait_item.id.owner_id.to_def_id())
                                    .self_ty()
                                    .skip_binder(),
                            );
                        }
                    }
                }
            }
        }

        let mut v = EscapeDelegate {
            cx,
            set: HirIdSet::default(),
            trait_self_ty,
            too_large_for_stack: self.too_large_for_stack,
        };

        let fn_def_id = cx.tcx.hir().local_def_id(hir_id);
        let infcx = cx.tcx.infer_ctxt().build();
        ExprUseVisitor::new(&mut v, &infcx, fn_def_id, cx.param_env, cx.typeck_results()).consume_body(body);

        for node in v.set {
            span_lint_hir(
                cx,
                BOXED_LOCAL,
                node,
                cx.tcx.hir().span(node),
                "local variable doesn't need to be boxed here",
            );
        }
    }
}

// TODO: Replace with Map::is_argument(..) when it's fixed
fn is_argument(map: rustc_middle::hir::map::Map<'_>, id: HirId) -> bool {
    match map.find(id) {
        Some(Node::Pat(Pat {
            kind: PatKind::Binding(..),
            ..
        })) => (),
        _ => return false,
    }

    matches!(map.find(map.get_parent_node(id)), Some(Node::Param(_)))
}

impl<'a, 'tcx> Delegate<'tcx> for EscapeDelegate<'a, 'tcx> {
    fn consume(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId) {
        if cmt.place.projections.is_empty() {
            if let PlaceBase::Local(lid) = cmt.place.base {
                self.set.remove(&lid);
            }
        }
    }

    fn borrow(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId, _: ty::BorrowKind) {
        if cmt.place.projections.is_empty() {
            if let PlaceBase::Local(lid) = cmt.place.base {
                self.set.remove(&lid);
            }
        }
    }

    fn mutate(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId) {
        if cmt.place.projections.is_empty() {
            let map = &self.cx.tcx.hir();
            if is_argument(*map, cmt.hir_id) {
                // Skip closure arguments
                let parent_id = map.get_parent_node(cmt.hir_id);
                if let Some(Node::Expr(..)) = map.find(map.get_parent_node(parent_id)) {
                    return;
                }

                // skip if there is a `self` parameter binding to a type
                // that contains `Self` (i.e.: `self: Box<Self>`), see #4804
                if let Some(trait_self_ty) = self.trait_self_ty {
                    if map.name(cmt.hir_id) == kw::SelfLower && cmt.place.ty().contains(trait_self_ty) {
                        return;
                    }
                }

                if is_non_trait_box(cmt.place.ty()) && !self.is_large_box(cmt.place.ty()) {
                    self.set.insert(cmt.hir_id);
                }
            }
        }
    }

    fn fake_read(&mut self, _: &rustc_hir_typeck::expr_use_visitor::PlaceWithHirId<'tcx>, _: FakeReadCause, _: HirId) {}
}

impl<'a, 'tcx> EscapeDelegate<'a, 'tcx> {
    fn is_large_box(&self, ty: Ty<'tcx>) -> bool {
        // Large types need to be boxed to avoid stack overflows.
        if ty.is_box() {
            self.cx.layout_of(ty.boxed_ty()).map_or(0, |l| l.size.bytes()) > self.too_large_for_stack
        } else {
            false
        }
    }
}
