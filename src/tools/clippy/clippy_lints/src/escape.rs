use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_hir;
use rustc_abi::ExternAbi;
use rustc_hir::{AssocItemKind, Body, FnDecl, HirId, HirIdSet, Impl, ItemKind, Node, Pat, PatKind, intravisit};
use rustc_hir_typeck::expr_use_visitor::{Delegate, ExprUseVisitor, PlaceBase, PlaceWithHirId};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, TraitRef, Ty, TyCtxt};
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::kw;

pub struct BoxedLocal {
    too_large_for_stack: u64,
}

impl BoxedLocal {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            too_large_for_stack: conf.too_large_for_stack,
        }
    }
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
    /// ```no_run
    /// fn foo(x: Box<u32>) {}
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// fn foo(x: u32) {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub BOXED_LOCAL,
    perf,
    "using `Box<T>` where unnecessary"
}

fn is_non_trait_box(ty: Ty<'_>) -> bool {
    ty.boxed_ty().is_some_and(|boxed| !boxed.is_trait())
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
        fn_def_id: LocalDefId,
    ) {
        if let Some(header) = fn_kind.header()
            && header.abi != ExternAbi::Rust
        {
            return;
        }

        let parent_id = cx
            .tcx
            .hir_get_parent_item(cx.tcx.local_def_id_to_hir_id(fn_def_id))
            .def_id;

        let mut trait_self_ty = None;
        if let Node::Item(item) = cx.tcx.hir_node_by_def_id(parent_id) {
            // If the method is an impl for a trait, don't warn.
            if let ItemKind::Impl(Impl { of_trait: Some(_), .. }) = item.kind {
                return;
            }

            // find `self` ty for this trait if relevant
            if let ItemKind::Trait(_, _, _, _, _, items) = item.kind {
                for trait_item in items {
                    if trait_item.id.owner_id.def_id == fn_def_id
                        // be sure we have `self` parameter in this function
                        && trait_item.kind == (AssocItemKind::Fn { has_self: true })
                    {
                        trait_self_ty = Some(TraitRef::identity(cx.tcx, trait_item.id.owner_id.to_def_id()).self_ty());
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

        ExprUseVisitor::for_clippy(cx, fn_def_id, &mut v)
            .consume_body(body)
            .into_ok();

        for node in v.set {
            span_lint_hir(
                cx,
                BOXED_LOCAL,
                node,
                cx.tcx.hir_span(node),
                "local variable doesn't need to be boxed here",
            );
        }
    }
}

// TODO: Replace with Map::is_argument(..) when it's fixed
fn is_argument(tcx: TyCtxt<'_>, id: HirId) -> bool {
    match tcx.hir_node(id) {
        Node::Pat(Pat {
            kind: PatKind::Binding(..),
            ..
        }) => (),
        _ => return false,
    }

    matches!(tcx.parent_hir_node(id), Node::Param(_))
}

impl<'tcx> Delegate<'tcx> for EscapeDelegate<'_, 'tcx> {
    fn consume(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId) {
        if cmt.place.projections.is_empty()
            && let PlaceBase::Local(lid) = cmt.place.base
        {
            // FIXME(rust/#120456) - is `swap_remove` correct?
            self.set.swap_remove(&lid);
        }
    }

    fn use_cloned(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn borrow(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId, _: ty::BorrowKind) {
        if cmt.place.projections.is_empty()
            && let PlaceBase::Local(lid) = cmt.place.base
        {
            // FIXME(rust/#120456) - is `swap_remove` correct?
            self.set.swap_remove(&lid);
        }
    }

    fn mutate(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId) {
        if cmt.place.projections.is_empty() && is_argument(self.cx.tcx, cmt.hir_id) {
            // Skip closure arguments
            let parent_id = self.cx.tcx.parent_hir_id(cmt.hir_id);
            if let Node::Expr(..) = self.cx.tcx.parent_hir_node(parent_id) {
                return;
            }

            // skip if there is a `self` parameter binding to a type
            // that contains `Self` (i.e.: `self: Box<Self>`), see #4804
            if let Some(trait_self_ty) = self.trait_self_ty
                && self.cx.tcx.hir_name(cmt.hir_id) == kw::SelfLower
                && cmt.place.ty().contains(trait_self_ty)
            {
                return;
            }

            if is_non_trait_box(cmt.place.ty()) && !self.is_large_box(cmt.place.ty()) {
                self.set.insert(cmt.hir_id);
            }
        }
    }

    fn fake_read(&mut self, _: &PlaceWithHirId<'tcx>, _: FakeReadCause, _: HirId) {}
}

impl<'tcx> EscapeDelegate<'_, 'tcx> {
    fn is_large_box(&self, ty: Ty<'tcx>) -> bool {
        // Large types need to be boxed to avoid stack overflows.
        if let Some(boxed_ty) = ty.boxed_ty() {
            self.cx.layout_of(boxed_ty).map_or(0, |l| l.size.bytes()) > self.too_large_for_stack
        } else {
            false
        }
    }
}
