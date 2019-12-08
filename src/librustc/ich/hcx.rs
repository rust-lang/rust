use crate::hir;
use crate::hir::def_id::{DefId, DefIndex};
use crate::hir::map::DefPathHash;
use crate::hir::map::definitions::Definitions;
use crate::ich::{self, CachingSourceMapView};
use crate::middle::cstore::CrateStore;
use crate::ty::{TyCtxt, fast_reject};
use crate::session::Session;

use std::cmp::Ord;

use syntax::ast;
use syntax::source_map::SourceMap;
use syntax::symbol::Symbol;
use syntax_pos::{SourceFile, BytePos};

use rustc_data_structures::stable_hasher::{
    HashStable, StableHasher, ToStableHashKey,
};
use rustc_data_structures::fx::{FxHashSet, FxHashMap};
use rustc_data_structures::sync::Lrc;
use smallvec::SmallVec;

fn compute_ignored_attr_names() -> FxHashSet<Symbol> {
    debug_assert!(ich::IGNORED_ATTRIBUTES.len() > 0);
    ich::IGNORED_ATTRIBUTES.iter().map(|&s| s).collect()
}

/// This is the context state available during incr. comp. hashing. It contains
/// enough information to transform `DefId`s and `HirId`s into stable `DefPath`s (i.e.,
/// a reference to the `TyCtxt`) and it holds a few caches for speeding up various
/// things (e.g., each `DefId`/`DefPath` is only hashed once).
#[derive(Clone)]
pub struct StableHashingContext<'a> {
    sess: &'a Session,
    definitions: &'a Definitions,
    cstore: &'a dyn CrateStore,
    body_resolver: BodyResolver<'a>,
    hash_spans: bool,
    hash_bodies: bool,
    node_id_hashing_mode: NodeIdHashingMode,

    // Very often, we are hashing something that does not need the
    // `CachingSourceMapView`, so we initialize it lazily.
    raw_source_map: &'a SourceMap,
    caching_source_map: Option<CachingSourceMapView<'a>>,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum NodeIdHashingMode {
    Ignore,
    HashDefPath,
}

/// The `BodyResolver` allows mapping a `BodyId` to the corresponding `hir::Body`.
/// We could also just store a plain reference to the `hir::Crate` but we want
/// to avoid that the crate is used to get untracked access to all of the HIR.
#[derive(Clone, Copy)]
struct BodyResolver<'tcx>(&'tcx hir::Crate);

impl<'tcx> BodyResolver<'tcx> {
    /// Returns a reference to the `hir::Body` with the given `BodyId`.
    /// **Does not do any tracking**; use carefully.
    fn body(self, id: hir::BodyId) -> &'tcx hir::Body {
        self.0.body(id)
    }
}

impl<'a> StableHashingContext<'a> {
    /// The `krate` here is only used for mapping `BodyId`s to `Body`s.
    /// Don't use it for anything else or you'll run the risk of
    /// leaking data out of the tracking system.
    #[inline]
    pub fn new(sess: &'a Session,
               krate: &'a hir::Crate,
               definitions: &'a Definitions,
               cstore: &'a dyn CrateStore)
               -> Self {
        let hash_spans_initial = !sess.opts.debugging_opts.incremental_ignore_spans;

        StableHashingContext {
            sess,
            body_resolver: BodyResolver(krate),
            definitions,
            cstore,
            caching_source_map: None,
            raw_source_map: sess.source_map(),
            hash_spans: hash_spans_initial,
            hash_bodies: true,
            node_id_hashing_mode: NodeIdHashingMode::HashDefPath,
        }
    }

    #[inline]
    pub fn sess(&self) -> &'a Session {
        self.sess
    }

    #[inline]
    pub fn while_hashing_hir_bodies<F: FnOnce(&mut Self)>(&mut self,
                                                          hash_bodies: bool,
                                                          f: F) {
        let prev_hash_bodies = self.hash_bodies;
        self.hash_bodies = hash_bodies;
        f(self);
        self.hash_bodies = prev_hash_bodies;
    }

    #[inline]
    pub fn while_hashing_spans<F: FnOnce(&mut Self)>(&mut self,
                                                     hash_spans: bool,
                                                     f: F) {
        let prev_hash_spans = self.hash_spans;
        self.hash_spans = hash_spans;
        f(self);
        self.hash_spans = prev_hash_spans;
    }

    #[inline]
    pub fn with_node_id_hashing_mode<F: FnOnce(&mut Self)>(&mut self,
                                                           mode: NodeIdHashingMode,
                                                           f: F) {
        let prev = self.node_id_hashing_mode;
        self.node_id_hashing_mode = mode;
        f(self);
        self.node_id_hashing_mode = prev;
    }

    #[inline]
    pub fn def_path_hash(&self, def_id: DefId) -> DefPathHash {
        if def_id.is_local() {
            self.definitions.def_path_hash(def_id.index)
        } else {
            self.cstore.def_path_hash(def_id)
        }
    }

    #[inline]
    pub fn local_def_path_hash(&self, def_index: DefIndex) -> DefPathHash {
        self.definitions.def_path_hash(def_index)
    }

    #[inline]
    pub fn node_to_hir_id(&self, node_id: ast::NodeId) -> hir::HirId {
        self.definitions.node_to_hir_id(node_id)
    }

    #[inline]
    pub fn hash_bodies(&self) -> bool {
        self.hash_bodies
    }

    #[inline]
    pub fn source_map(&mut self) -> &mut CachingSourceMapView<'a> {
        match self.caching_source_map {
            Some(ref mut cm) => {
                cm
            }
            ref mut none => {
                *none = Some(CachingSourceMapView::new(self.raw_source_map));
                none.as_mut().unwrap()
            }
        }
    }

    #[inline]
    pub fn is_ignored_attr(&self, name: Symbol) -> bool {
        thread_local! {
            static IGNORED_ATTRIBUTES: FxHashSet<Symbol> = compute_ignored_attr_names();
        }
        IGNORED_ATTRIBUTES.with(|attrs| attrs.contains(&name))
    }

    pub fn hash_hir_item_like<F: FnOnce(&mut Self)>(&mut self, f: F) {
        let prev_hash_node_ids = self.node_id_hashing_mode;
        self.node_id_hashing_mode = NodeIdHashingMode::Ignore;

        f(self);

        self.node_id_hashing_mode = prev_hash_node_ids;
    }
}

/// Something that can provide a stable hashing context.
pub trait StableHashingContextProvider<'a> {
    fn get_stable_hashing_context(&self) -> StableHashingContext<'a>;
}

impl<'a, 'b, T: StableHashingContextProvider<'a>> StableHashingContextProvider<'a>
for &'b T {
    fn get_stable_hashing_context(&self) -> StableHashingContext<'a> {
        (**self).get_stable_hashing_context()
    }
}

impl<'a, 'b, T: StableHashingContextProvider<'a>> StableHashingContextProvider<'a>
for &'b mut T {
    fn get_stable_hashing_context(&self) -> StableHashingContext<'a> {
        (**self).get_stable_hashing_context()
    }
}

impl StableHashingContextProvider<'tcx> for TyCtxt<'tcx> {
    fn get_stable_hashing_context(&self) -> StableHashingContext<'tcx> {
        (*self).create_stable_hashing_context()
    }
}

impl<'a> StableHashingContextProvider<'a> for StableHashingContext<'a> {
    fn get_stable_hashing_context(&self) -> StableHashingContext<'a> {
        self.clone()
    }
}

impl<'a> crate::dep_graph::DepGraphSafe for StableHashingContext<'a> {}

impl<'a> HashStable<StableHashingContext<'a>> for hir::BodyId {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        if hcx.hash_bodies() {
            hcx.body_resolver.body(*self).hash_stable(hcx, hasher);
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::HirId {
    #[inline]
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        match hcx.node_id_hashing_mode {
            NodeIdHashingMode::Ignore => {
                // Don't do anything.
            }
            NodeIdHashingMode::HashDefPath => {
                let hir::HirId {
                    owner,
                    local_id,
                } = *self;

                hcx.local_def_path_hash(owner).hash_stable(hcx, hasher);
                local_id.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for hir::HirId {
    type KeyType = (DefPathHash, hir::ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self,
                          hcx: &StableHashingContext<'a>)
                          -> (DefPathHash, hir::ItemLocalId) {
        let def_path_hash = hcx.local_def_path_hash(self.owner);
        (def_path_hash, self.local_id)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ast::NodeId {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        match hcx.node_id_hashing_mode {
            NodeIdHashingMode::Ignore => {
                // Don't do anything.
            }
            NodeIdHashingMode::HashDefPath => {
                hcx.definitions.node_to_hir_id(*self).hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for ast::NodeId {
    type KeyType = (DefPathHash, hir::ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self,
                          hcx: &StableHashingContext<'a>)
                          -> (DefPathHash, hir::ItemLocalId) {
        hcx.definitions.node_to_hir_id(*self).to_stable_hash_key(hcx)
    }
}

impl<'a> syntax_pos::HashStableContext for StableHashingContext<'a> {
    fn hash_spans(&self) -> bool {
        self.hash_spans
    }

    fn byte_pos_to_line_and_col(&mut self, byte: BytePos)
        -> Option<(Lrc<SourceFile>, usize, BytePos)>
    {
        self.source_map().byte_pos_to_line_and_col(byte)
    }
}

pub fn hash_stable_trait_impls<'a>(
    hcx: &mut StableHashingContext<'a>,
    hasher: &mut StableHasher,
    blanket_impls: &[DefId],
    non_blanket_impls: &FxHashMap<fast_reject::SimplifiedType, Vec<DefId>>,
) {
    {
        let mut blanket_impls: SmallVec<[_; 8]> = blanket_impls
            .iter()
            .map(|&def_id| hcx.def_path_hash(def_id))
            .collect();

        if blanket_impls.len() > 1 {
            blanket_impls.sort_unstable();
        }

        blanket_impls.hash_stable(hcx, hasher);
    }

    {
        let mut keys: SmallVec<[_; 8]> =
            non_blanket_impls.keys()
                             .map(|k| (k, k.map_def(|d| hcx.def_path_hash(d))))
                             .collect();
        keys.sort_unstable_by(|&(_, ref k1), &(_, ref k2)| k1.cmp(k2));
        keys.len().hash_stable(hcx, hasher);
        for (key, ref stable_key) in keys {
            stable_key.hash_stable(hcx, hasher);
            let mut impls : SmallVec<[_; 8]> = non_blanket_impls[key]
                .iter()
                .map(|&impl_id| hcx.def_path_hash(impl_id))
                .collect();

            if impls.len() > 1 {
                impls.sort_unstable();
            }

            impls.hash_stable(hcx, hasher);
        }
    }
}
