use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;

fn is_parent_const_impl_raw(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    let parent_id = tcx.local_parent(def_id);
    matches!(tcx.def_kind(parent_id), DefKind::Impl { .. })
        && tcx.constness(parent_id) == hir::Constness::Const
}

/// Checks whether an item is considered to be `const`. If it is a constructor, anonymous const,
/// const block, const item or associated const, it is const. If it is a trait impl/function,
/// return if it has a `const` modifier. If it is an intrinsic, report whether said intrinsic
/// has a `rustc_const_{un,}stable` attribute. Otherwise, return `Constness::NotConst`.
fn constness(tcx: TyCtxt<'_>, def_id: LocalDefId) -> hir::Constness {
    let node = tcx.hir_node_by_def_id(def_id);

    match node {
        hir::Node::Ctor(hir::VariantData::Tuple(..))
        | hir::Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Const(..), .. }) => {
            hir::Constness::Const
        }
        hir::Node::Item(hir::Item { kind: hir::ItemKind::Impl(impl_), .. }) => impl_.constness,
        hir::Node::ForeignItem(_) => {
            // Foreign items cannot be evaluated at compile-time.
            hir::Constness::NotConst
        }
        hir::Node::Expr(e) if let hir::ExprKind::Closure(c) = e.kind => c.constness,
        _ => {
            if let Some(fn_kind) = node.fn_kind() {
                if fn_kind.constness() == hir::Constness::Const {
                    return hir::Constness::Const;
                }

                // If the function itself is not annotated with `const`, it may still be a `const fn`
                // if it resides in a const trait impl.
                let is_const = is_parent_const_impl_raw(tcx, def_id);
                if is_const { hir::Constness::Const } else { hir::Constness::NotConst }
            } else {
                tcx.dcx().span_bug(
                    tcx.def_span(def_id),
                    format!("should not be requesting the constness of items that can't be const: {node:#?}: {:?}", tcx.def_kind(def_id))
                )
            }
        }
    }
}

fn is_promotable_const_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.is_const_fn(def_id)
        && match tcx.lookup_const_stability(def_id) {
            Some(stab) => {
                if cfg!(debug_assertions) && stab.promotable {
                    let sig = tcx.fn_sig(def_id);
                    assert_eq!(
                        sig.skip_binder().safety(),
                        hir::Safety::Safe,
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
