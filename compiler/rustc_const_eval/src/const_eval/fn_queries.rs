use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::hir::map::blocks::FnLikeNode;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::Symbol;
use rustc_target::spec::abi::Abi;

/// Whether the `def_id` counts as const fn in your current crate, considering all active
/// feature gates
pub fn is_const_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.is_const_fn_raw(def_id)
        && match is_unstable_const_fn(tcx, def_id) {
            Some(feature_name) => {
                // has a `rustc_const_unstable` attribute, check whether the user enabled the
                // corresponding feature gate.
                tcx.features().declared_lib_features.iter().any(|&(sym, _)| sym == feature_name)
            }
            // functions without const stability are either stable user written
            // const fn or the user is using feature gates and we thus don't
            // care what they do
            None => true,
        }
}

/// Whether the `def_id` is an unstable const fn and what feature gate is necessary to enable it
pub fn is_unstable_const_fn(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Symbol> {
    if tcx.is_const_fn_raw(def_id) {
        let const_stab = tcx.lookup_const_stability(def_id)?;
        if const_stab.level.is_unstable() { Some(const_stab.feature) } else { None }
    } else {
        None
    }
}

pub fn is_parent_const_impl_raw(tcx: TyCtxt<'_>, hir_id: hir::HirId) -> bool {
    let parent_id = tcx.hir().get_parent_node(hir_id);
    matches!(
        tcx.hir().get(parent_id),
        hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Impl(hir::Impl { constness: hir::Constness::Const, .. }),
            ..
        })
    )
}

/// Checks whether the function has a `const` modifier or, in case it is an intrinsic, whether
/// said intrinsic has a `rustc_const_{un,}stable` attribute.
fn is_const_fn_raw(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());

    let node = tcx.hir().get(hir_id);

    if let hir::Node::ForeignItem(hir::ForeignItem { kind: hir::ForeignItemKind::Fn(..), .. }) =
        node
    {
        // Intrinsics use `rustc_const_{un,}stable` attributes to indicate constness. All other
        // foreign items cannot be evaluated at compile-time.
        if let Abi::RustIntrinsic | Abi::PlatformIntrinsic = tcx.hir().get_foreign_abi(hir_id) {
            tcx.lookup_const_stability(def_id).is_some()
        } else {
            false
        }
    } else if let Some(fn_like) = FnLikeNode::from_node(node) {
        if fn_like.constness() == hir::Constness::Const {
            return true;
        }

        // If the function itself is not annotated with `const`, it may still be a `const fn`
        // if it resides in a const trait impl.
        is_parent_const_impl_raw(tcx, hir_id)
    } else if let hir::Node::Ctor(_) = node {
        true
    } else {
        false
    }
}

fn is_promotable_const_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    is_const_fn(tcx, def_id)
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
    *providers = Providers { is_const_fn_raw, is_promotable_const_fn, ..*providers };
}
