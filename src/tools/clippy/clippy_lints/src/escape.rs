use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_hir;
use rustc_abi::ExternAbi;
use rustc_hir::def::DefKind;
use rustc_hir::{Body, FnDecl, HirId, HirIdSet, PatKind, intravisit};
use rustc_hir_typeck::expr_use_visitor::{Delegate, ExprUseVisitor, PlaceBase, PlaceWithHirId};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::layout::LayoutOf as _;
use rustc_middle::ty::{self, TraitRef, Ty};
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

impl_lint_pass!(BoxedLocal => [BOXED_LOCAL]);

/// Whether `ty` is a `Box<T>` where `T` is not a trait object and is small enough to live on the
/// stack (large types are boxed to avoid stack overflows).
fn is_small_non_trait_box<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, too_large_for_stack: u64) -> bool {
    ty.boxed_ty().is_some_and(|boxed| {
        !boxed.is_trait() && cx.layout_of(boxed).map_or(0, |l| l.size.bytes()) <= too_large_for_stack
    })
}

struct EscapeDelegate {
    set: HirIdSet,
}

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
        // Skip closures
        if matches!(fn_kind, intravisit::FnKind::Closure) {
            return;
        }

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
        match cx.tcx.def_kind(parent_id) {
            // If the method is an impl for a trait, don't warn.
            DefKind::Impl { of_trait: true } => return,

            // find `self` ty for this trait if relevant
            DefKind::Trait => {
                trait_self_ty = Some(TraitRef::identity(cx.tcx, parent_id.to_def_id()).self_ty());
            },

            _ => {},
        }

        let typeck_results = cx.tcx.typeck_body(body.id());

        // Seed the set with the `Box` parameters that could be unboxed. The `ExprUseVisitor` walk
        // below then removes any that escape by being moved or borrowed.
        let set: HirIdSet = body
            .params
            .iter()
            .filter_map(|param| {
                // Only simple bindings (`x: Box<_>`) bind a local that the walk can track and report.
                if !matches!(param.pat.kind, PatKind::Binding(..)) {
                    return None;
                }

                let ty = typeck_results.pat_ty(param.pat);
                if !is_small_non_trait_box(cx, ty, self.too_large_for_stack) {
                    return None;
                }

                // skip `self` parameters whose type contains `Self` (i.e.: `self: Box<Self>`), see #4804
                if let Some(trait_self_ty) = trait_self_ty
                    && cx.tcx.hir_name(param.pat.hir_id) == kw::SelfLower
                    && ty.contains(trait_self_ty)
                {
                    return None;
                }

                Some(param.pat.hir_id)
            })
            .collect();

        // Without any candidate parameter, the expensive `ExprUseVisitor` walk can never lint.
        if set.is_empty() {
            return;
        }

        let mut v = EscapeDelegate { set };

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

impl<'tcx> Delegate<'tcx> for EscapeDelegate {
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

    fn mutate(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn fake_read(&mut self, _: &PlaceWithHirId<'tcx>, _: FakeReadCause, _: HirId) {}
}
