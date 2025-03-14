use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;

fn parent_impl_or_trait_constness(tcx: TyCtxt<'_>, def_id: LocalDefId) -> hir::Constness {
    let parent_id = tcx.local_parent(def_id);
    match tcx.def_kind(parent_id) {
        DefKind::Impl { of_trait: true } => tcx.impl_trait_header(parent_id).unwrap().constness,
        DefKind::Trait => {
            if tcx.is_const_trait(parent_id.into()) {
                hir::Constness::Const
            } else {
                hir::Constness::NotConst
            }
        }
        _ => hir::Constness::NotConst,
    }
}

/// Checks whether a function-like definition is considered to be `const`.
fn constness(tcx: TyCtxt<'_>, def_id: LocalDefId) -> hir::Constness {
    let node = tcx.hir_node_by_def_id(def_id);

    match node {
        hir::Node::Ctor(hir::VariantData::Tuple(..)) => hir::Constness::Const,
        hir::Node::ForeignItem(item) if let hir::ForeignItemKind::Fn(..) = item.kind => {
            // Foreign functions cannot be evaluated at compile-time.
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
                parent_impl_or_trait_constness(tcx, def_id)
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
