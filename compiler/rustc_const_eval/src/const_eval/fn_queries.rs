use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{DefIdTree, TyCtxt};
use rustc_span::symbol::Symbol;

/// Whether the `def_id` is an unstable const fn and what feature gate is necessary to enable it
pub fn is_unstable_const_fn(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Symbol> {
    if tcx.is_const_fn_raw(def_id) {
        let const_stab = tcx.lookup_const_stability(def_id)?;
        if const_stab.is_const_unstable() { Some(const_stab.feature) } else { None }
    } else {
        None
    }
}

pub fn is_parent_const_impl_raw(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    let parent_id = tcx.local_parent(def_id);
    tcx.def_kind(parent_id) == DefKind::Impl && tcx.constness(parent_id) == hir::Constness::Const
}

/// Checks whether an item is considered to be `const`. If it is a constructor, it is const. If
/// it is a trait impl/function, return if it has a `const` modifier. If it is an intrinsic,
/// report whether said intrinsic has a `rustc_const_{un,}stable` attribute. Otherwise, return
/// `Constness::NotConst`.
fn constness(tcx: TyCtxt<'_>, def_id: DefId) -> hir::Constness {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    match tcx.hir().get(hir_id) {
        hir::Node::Ctor(_) => hir::Constness::Const,

        hir::Node::ForeignItem(hir::ForeignItem { kind: hir::ForeignItemKind::Fn(..), .. }) => {
            // Intrinsics use `rustc_const_{un,}stable` attributes to indicate constness. All other
            // foreign items cannot be evaluated at compile-time.
            let is_const = if tcx.is_intrinsic(def_id) {
                tcx.lookup_const_stability(def_id).is_some()
            } else {
                false
            };
            if is_const { hir::Constness::Const } else { hir::Constness::NotConst }
        }

        hir::Node::TraitItem(hir::TraitItem { kind: hir::TraitItemKind::Fn(..), .. })
            if tcx.is_const_default_method(def_id) =>
        {
            hir::Constness::Const
        }

        hir::Node::Item(hir::Item { kind: hir::ItemKind::Const(..), .. })
        | hir::Node::Item(hir::Item { kind: hir::ItemKind::Static(..), .. })
        | hir::Node::TraitItem(hir::TraitItem { kind: hir::TraitItemKind::Const(..), .. })
        | hir::Node::AnonConst(_)
        | hir::Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Const(..), .. })
        | hir::Node::ImplItem(hir::ImplItem {
            kind:
                hir::ImplItemKind::Fn(
                    hir::FnSig {
                        header: hir::FnHeader { constness: hir::Constness::Const, .. },
                        ..
                    },
                    ..,
                ),
            ..
        }) => hir::Constness::Const,

        hir::Node::ImplItem(hir::ImplItem {
            kind: hir::ImplItemKind::Type(..) | hir::ImplItemKind::Fn(..),
            ..
        }) => {
            let parent_hir_id = tcx.hir().get_parent_node(hir_id);
            match tcx.hir().get(parent_hir_id) {
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Impl(hir::Impl { constness, .. }),
                    ..
                }) => *constness,
                _ => span_bug!(
                    tcx.def_span(parent_hir_id.owner),
                    "impl item's parent node is not an impl",
                ),
            }
        }

        hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Fn(hir::FnSig { header: hir::FnHeader { constness, .. }, .. }, ..),
            ..
        })
        | hir::Node::TraitItem(hir::TraitItem {
            kind:
                hir::TraitItemKind::Fn(hir::FnSig { header: hir::FnHeader { constness, .. }, .. }, ..),
            ..
        })
        | hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Impl(hir::Impl { constness, .. }),
            ..
        }) => *constness,

        _ => hir::Constness::NotConst,
    }
}

fn is_promotable_const_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.is_const_fn(def_id)
        && match tcx.lookup_const_stability(def_id) {
            Some(stab) => {
                if cfg!(debug_assertions) && stab.promotable {
                    let sig = tcx.fn_sig(def_id);
                    assert_eq!(
                        sig.unsafety(),
                        hir::Unsafety::Normal,
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
