use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{
    Constness, ExprKind, ForeignItemKind, ImplItem, ImplItemImplKind, ImplItemKind, Item, ItemKind,
    Node, TraitItem, TraitItemKind, VariantData,
};
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;

/// Checks whether a function-like definition is considered to be `const`. Also stores constness of inherent impls.
fn constness(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Constness {
    let node = tcx.hir_node_by_def_id(def_id);

    match node {
        Node::Ctor(VariantData::Tuple(..)) => Constness::Const,
        Node::ForeignItem(item) if let ForeignItemKind::Fn(..) = item.kind => {
            // Foreign functions cannot be evaluated at compile-time.
            Constness::NotConst
        }
        Node::Expr(e) if let ExprKind::Closure(c) = e.kind => c.constness,
        // FIXME(fee1-dead): extract this one out and rename this query to `fn_constness` so we don't need `is_const_fn` anymore.
        Node::Item(i) if let ItemKind::Impl(impl_) = i.kind => impl_.constness,
        Node::Item(Item { kind: ItemKind::Fn { sig, .. }, .. }) => sig.header.constness,
        Node::ImplItem(ImplItem {
            impl_kind: ImplItemImplKind::Trait { .. },
            kind: ImplItemKind::Fn(..),
            ..
        }) => tcx.impl_trait_header(tcx.local_parent(def_id)).constness,
        Node::ImplItem(ImplItem {
            impl_kind: ImplItemImplKind::Inherent { .. },
            kind: ImplItemKind::Fn(sig, _),
            ..
        }) => {
            match sig.header.constness {
                Constness::Const => Constness::Const,
                // inherent impl could be const
                Constness::NotConst => tcx.constness(tcx.local_parent(def_id)),
            }
        }
        Node::TraitItem(TraitItem { kind: TraitItemKind::Fn(..), .. }) => tcx.trait_def(tcx.local_parent(def_id)).constness,
        _ => {
            tcx.dcx().span_bug(
                tcx.def_span(def_id),
                format!("should not be requesting the constness of items that can't be const: {node:#?}: {:?}", tcx.def_kind(def_id))
            )
        }
    }
}

fn is_promotable_const_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.is_const_fn(def_id)
        && match tcx.lookup_const_stability(def_id) {
            Some(stab) => {
                if cfg!(debug_assertions) && stab.promotable {
                    let sig = tcx.fn_sig(def_id);
                    assert!(
                        sig.skip_binder().safety().is_safe(),
                        "don't mark const unsafe fns as promotable",
                        // https://github.com/rust-lang/rust/pull/53851#issuecomment-418760682
                    );
                }
                stab.promotable
            }
            None => false,
        }
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { constness, is_promotable_const_fn, ..*providers };
}
