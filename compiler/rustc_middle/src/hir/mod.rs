//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

pub mod exports;
pub mod map;
pub mod place;

use crate::ty::query::Providers;
use crate::ty::TyCtxt;
use rustc_ast::Attribute;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::*;
use rustc_index::vec::{Idx, IndexVec};
use rustc_query_system::ich::StableHashingContext;
use rustc_span::DUMMY_SP;
use std::collections::BTreeMap;

/// Result of HIR indexing.
#[derive(Debug)]
pub struct IndexedHir<'hir> {
    /// Contents of the HIR owned by each definition. None for definitions that are not HIR owners.
    // The `mut` comes from construction time, and is harmless since we only ever hand out
    // immutable refs to IndexedHir.
    map: IndexVec<LocalDefId, Option<&'hir mut OwnerNodes<'hir>>>,
    /// Map from each owner to its parent's HirId inside another owner.
    // This map is separate from `map` to eventually allow for per-owner indexing.
    parenting: FxHashMap<LocalDefId, HirId>,
}

/// Top-level HIR node for current owner. This only contains the node for which
/// `HirId::local_id == 0`, and excludes bodies.
///
/// This struct exists to encapsulate all access to the hir_owner query in this module, and to
/// implement HashStable without hashing bodies.
#[derive(Copy, Clone, Debug)]
pub struct Owner<'tcx> {
    node: OwnerNode<'tcx>,
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for Owner<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let Owner { node } = self;
        hcx.while_hashing_hir_bodies(false, |hcx| node.hash_stable(hcx, hasher));
    }
}

/// HIR node coupled with its parent's id in the same HIR owner.
///
/// The parent is trash when the node is a HIR owner.
#[derive(Clone, Debug)]
pub struct ParentedNode<'tcx> {
    parent: ItemLocalId,
    node: Node<'tcx>,
}

#[derive(Debug)]
pub struct OwnerNodes<'tcx> {
    /// Pre-computed hash of the full HIR.
    hash: Fingerprint,
    /// Full HIR for the current owner.
    // The zeroth node's parent is trash, but is never accessed.
    nodes: IndexVec<ItemLocalId, Option<ParentedNode<'tcx>>>,
    /// Content of local bodies.
    bodies: FxHashMap<ItemLocalId, &'tcx Body<'tcx>>,
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for OwnerNodes<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        // We ignore the `nodes` and `bodies` fields since these refer to information included in
        // `hash` which is hashed in the collector and used for the crate hash.
        let OwnerNodes { hash, nodes: _, bodies: _ } = *self;
        hash.hash_stable(hcx, hasher);
    }
}

/// Attributes owner by a HIR owner. It is build as a slice inside the attributes map, restricted
/// to the nodes whose `HirId::owner` is `prefix`.
#[derive(Copy, Clone)]
pub struct AttributeMap<'tcx> {
    map: &'tcx BTreeMap<HirId, &'tcx [Attribute]>,
    prefix: LocalDefId,
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for AttributeMap<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let range = self.range();

        range.clone().count().hash_stable(hcx, hasher);
        for (key, value) in range {
            key.hash_stable(hcx, hasher);
            value.hash_stable(hcx, hasher);
        }
    }
}

impl<'tcx> std::fmt::Debug for AttributeMap<'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttributeMap")
            .field("prefix", &self.prefix)
            .field("range", &&self.range().collect::<Vec<_>>()[..])
            .finish()
    }
}

impl<'tcx> AttributeMap<'tcx> {
    fn get(&self, id: ItemLocalId) -> &'tcx [Attribute] {
        self.map.get(&HirId { owner: self.prefix, local_id: id }).copied().unwrap_or(&[])
    }

    fn range(&self) -> std::collections::btree_map::Range<'_, rustc_hir::HirId, &[Attribute]> {
        let local_zero = ItemLocalId::from_u32(0);
        let range = HirId { owner: self.prefix, local_id: local_zero }..HirId {
            owner: LocalDefId { local_def_index: self.prefix.local_def_index + 1 },
            local_id: local_zero,
        };
        self.map.range(range)
    }
}

/// Gather the LocalDefId for each item-like within a module, including items contained within
/// bodies.  The Ids are in visitor order.  This is used to partition a pass between modules.
#[derive(Debug, HashStable)]
pub struct ModuleItems {
    submodules: Box<[LocalDefId]>,
    items: Box<[ItemId]>,
    trait_items: Box<[TraitItemId]>,
    impl_items: Box<[ImplItemId]>,
    foreign_items: Box<[ForeignItemId]>,
}

impl<'tcx> TyCtxt<'tcx> {
    #[inline(always)]
    pub fn hir(self) -> map::Map<'tcx> {
        map::Map { tcx: self }
    }

    pub fn parent_module(self, id: HirId) -> LocalDefId {
        self.parent_module_from_def_id(id.owner)
    }
}

pub fn provide(providers: &mut Providers) {
    providers.parent_module_from_def_id = |tcx, id| {
        let hir = tcx.hir();
        hir.local_def_id(hir.get_module_parent_node(hir.local_def_id_to_hir_id(id)))
    };
    providers.hir_crate = |tcx, ()| tcx.untracked_crate;
    providers.index_hir = map::index_hir;
    providers.crate_hash = map::crate_hash;
    providers.hir_module_items = map::hir_module_items;
    providers.hir_owner = |tcx, id| {
        let owner = tcx.index_hir(()).map[id].as_ref()?;
        let node = owner.nodes[ItemLocalId::new(0)].as_ref().unwrap().node;
        let node = node.as_owner().unwrap(); // Indexing must ensure it is an OwnerNode.
        Some(Owner { node })
    };
    providers.hir_owner_nodes = |tcx, id| tcx.index_hir(()).map[id].as_deref();
    providers.hir_owner_parent = |tcx, id| {
        let index = tcx.index_hir(());
        index.parenting.get(&id).copied().unwrap_or(CRATE_HIR_ID)
    };
    providers.hir_attrs = |tcx, id| AttributeMap { map: &tcx.untracked_crate.attrs, prefix: id };
    providers.source_span = |tcx, def_id| tcx.resolutions(()).definitions.def_span(def_id);
    providers.def_span = |tcx, def_id| tcx.hir().span_if_local(def_id).unwrap_or(DUMMY_SP);
    providers.fn_arg_names = |tcx, id| {
        let hir = tcx.hir();
        let hir_id = hir.local_def_id_to_hir_id(id.expect_local());
        if let Some(body_id) = hir.maybe_body_owned_by(hir_id) {
            tcx.arena.alloc_from_iter(hir.body_param_names(body_id))
        } else if let Node::TraitItem(&TraitItem {
            kind: TraitItemKind::Fn(_, TraitFn::Required(idents)),
            ..
        }) = hir.get(hir_id)
        {
            tcx.arena.alloc_slice(idents)
        } else {
            span_bug!(hir.span(hir_id), "fn_arg_names: unexpected item {:?}", id);
        }
    };
    providers.opt_def_kind = |tcx, def_id| tcx.hir().opt_def_kind(def_id.expect_local());
    providers.all_local_trait_impls = |tcx, ()| &tcx.resolutions(()).trait_impls;
    providers.expn_that_defined = |tcx, id| {
        let id = id.expect_local();
        tcx.resolutions(()).definitions.expansion_that_defined(id)
    };
}
