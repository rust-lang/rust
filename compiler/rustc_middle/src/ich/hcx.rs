use crate::ich;
use crate::middle::cstore::CrateStore;
use crate::ty::{fast_reject, TyCtxt};

use rustc_ast as ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::Lrc;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::definitions::{DefPathHash, Definitions};
use rustc_session::Session;
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::Symbol;
use rustc_span::{BytePos, CachingSourceMapView, SourceFile, SpanData};

use rustc_span::def_id::{CrateNum, CRATE_DEF_INDEX};
use smallvec::SmallVec;
use std::cmp::Ord;

fn compute_ignored_attr_names() -> FxHashSet<Symbol> {
    debug_assert!(!ich::IGNORED_ATTRIBUTES.is_empty());
    ich::IGNORED_ATTRIBUTES.iter().copied().collect()
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
    pub(super) body_resolver: BodyResolver<'a>,
    hash_spans: bool,
    hash_bodies: bool,
    pub(super) node_id_hashing_mode: NodeIdHashingMode,

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
pub(super) struct BodyResolver<'tcx>(&'tcx hir::Crate<'tcx>);

impl<'tcx> BodyResolver<'tcx> {
    /// Returns a reference to the `hir::Body` with the given `BodyId`.
    /// **Does not do any tracking**; use carefully.
    pub(super) fn body(self, id: hir::BodyId) -> &'tcx hir::Body<'tcx> {
        self.0.body(id)
    }
}

impl<'a> StableHashingContext<'a> {
    /// The `krate` here is only used for mapping `BodyId`s to `Body`s.
    /// Don't use it for anything else or you'll run the risk of
    /// leaking data out of the tracking system.
    #[inline]
    fn new_with_or_without_spans(
        sess: &'a Session,
        krate: &'a hir::Crate<'a>,
        definitions: &'a Definitions,
        cstore: &'a dyn CrateStore,
        always_ignore_spans: bool,
    ) -> Self {
        let hash_spans_initial =
            !always_ignore_spans && !sess.opts.debugging_opts.incremental_ignore_spans;

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
    pub fn new(
        sess: &'a Session,
        krate: &'a hir::Crate<'a>,
        definitions: &'a Definitions,
        cstore: &'a dyn CrateStore,
    ) -> Self {
        Self::new_with_or_without_spans(
            sess,
            krate,
            definitions,
            cstore,
            /*always_ignore_spans=*/ false,
        )
    }

    #[inline]
    pub fn ignore_spans(
        sess: &'a Session,
        krate: &'a hir::Crate<'a>,
        definitions: &'a Definitions,
        cstore: &'a dyn CrateStore,
    ) -> Self {
        let always_ignore_spans = true;
        Self::new_with_or_without_spans(sess, krate, definitions, cstore, always_ignore_spans)
    }

    #[inline]
    pub fn sess(&self) -> &'a Session {
        self.sess
    }

    #[inline]
    pub fn while_hashing_hir_bodies<F: FnOnce(&mut Self)>(&mut self, hash_bodies: bool, f: F) {
        let prev_hash_bodies = self.hash_bodies;
        self.hash_bodies = hash_bodies;
        f(self);
        self.hash_bodies = prev_hash_bodies;
    }

    #[inline]
    pub fn while_hashing_spans<F: FnOnce(&mut Self)>(&mut self, hash_spans: bool, f: F) {
        let prev_hash_spans = self.hash_spans;
        self.hash_spans = hash_spans;
        f(self);
        self.hash_spans = prev_hash_spans;
    }

    #[inline]
    pub fn with_node_id_hashing_mode<F: FnOnce(&mut Self)>(
        &mut self,
        mode: NodeIdHashingMode,
        f: F,
    ) {
        let prev = self.node_id_hashing_mode;
        self.node_id_hashing_mode = mode;
        f(self);
        self.node_id_hashing_mode = prev;
    }

    #[inline]
    pub fn def_path_hash(&self, def_id: DefId) -> DefPathHash {
        if let Some(def_id) = def_id.as_local() {
            self.local_def_path_hash(def_id)
        } else {
            self.cstore.def_path_hash(def_id)
        }
    }

    #[inline]
    pub fn local_def_path_hash(&self, def_id: LocalDefId) -> DefPathHash {
        self.definitions.def_path_hash(def_id)
    }

    #[inline]
    pub fn hash_bodies(&self) -> bool {
        self.hash_bodies
    }

    #[inline]
    pub fn source_map(&mut self) -> &mut CachingSourceMapView<'a> {
        match self.caching_source_map {
            Some(ref mut sm) => sm,
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
}

/// Something that can provide a stable hashing context.
pub trait StableHashingContextProvider<'a> {
    fn get_stable_hashing_context(&self) -> StableHashingContext<'a>;
}

impl<'a, 'b, T: StableHashingContextProvider<'a>> StableHashingContextProvider<'a> for &'b T {
    fn get_stable_hashing_context(&self) -> StableHashingContext<'a> {
        (**self).get_stable_hashing_context()
    }
}

impl<'a, 'b, T: StableHashingContextProvider<'a>> StableHashingContextProvider<'a> for &'b mut T {
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

impl<'a> HashStable<StableHashingContext<'a>> for ast::NodeId {
    fn hash_stable(&self, _: &mut StableHashingContext<'a>, _: &mut StableHasher) {
        panic!("Node IDs should not appear in incremental state");
    }
}

impl<'a> rustc_span::HashStableContext for StableHashingContext<'a> {
    fn hash_spans(&self) -> bool {
        self.hash_spans
    }

    #[inline]
    fn hash_crate_num(&mut self, cnum: CrateNum, hasher: &mut StableHasher) {
        let hcx = self;
        hcx.def_path_hash(DefId { krate: cnum, index: CRATE_DEF_INDEX }).hash_stable(hcx, hasher);
    }

    #[inline]
    fn hash_def_id(&mut self, def_id: DefId, hasher: &mut StableHasher) {
        let hcx = self;
        hcx.def_path_hash(def_id).hash_stable(hcx, hasher);
    }

    fn byte_pos_to_line_and_col(
        &mut self,
        byte: BytePos,
    ) -> Option<(Lrc<SourceFile>, usize, BytePos)> {
        self.source_map().byte_pos_to_line_and_col(byte)
    }

    fn span_data_to_lines_and_cols(
        &mut self,
        span: &SpanData,
    ) -> Option<(Lrc<SourceFile>, usize, BytePos, usize, BytePos)> {
        self.source_map().span_data_to_lines_and_cols(span)
    }
}

pub fn hash_stable_trait_impls<'a>(
    hcx: &mut StableHashingContext<'a>,
    hasher: &mut StableHasher,
    blanket_impls: &[DefId],
    non_blanket_impls: &FxHashMap<fast_reject::SimplifiedType, Vec<DefId>>,
) {
    {
        let mut blanket_impls: SmallVec<[_; 8]> =
            blanket_impls.iter().map(|&def_id| hcx.def_path_hash(def_id)).collect();

        if blanket_impls.len() > 1 {
            blanket_impls.sort_unstable();
        }

        blanket_impls.hash_stable(hcx, hasher);
    }

    {
        let mut keys: SmallVec<[_; 8]> =
            non_blanket_impls.keys().map(|k| (k, k.map_def(|d| hcx.def_path_hash(d)))).collect();
        keys.sort_unstable_by(|&(_, ref k1), &(_, ref k2)| k1.cmp(k2));
        keys.len().hash_stable(hcx, hasher);
        for (key, ref stable_key) in keys {
            stable_key.hash_stable(hcx, hasher);
            let mut impls: SmallVec<[_; 8]> =
                non_blanket_impls[key].iter().map(|&impl_id| hcx.def_path_hash(impl_id)).collect();

            if impls.len() > 1 {
                impls.sort_unstable();
            }

            impls.hash_stable(hcx, hasher);
        }
    }
}
