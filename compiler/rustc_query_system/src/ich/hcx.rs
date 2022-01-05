use crate::ich;
use rustc_ast as ast;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sorted_map::SortedMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::stable_hasher::{HashingControls, NodeIdHashingMode};
use rustc_data_structures::sync::Lrc;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::definitions::{DefPathHash, Definitions};
use rustc_session::cstore::CrateStore;
use rustc_session::Session;
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::Symbol;
use rustc_span::{BytePos, CachingSourceMapView, SourceFile, Span, SpanData};

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
    definitions: &'a Definitions,
    cstore: &'a dyn CrateStore,
    // The value of `-Z incremental-ignore-spans`.
    // This field should only be used by `debug_opts_incremental_ignore_span`
    incremental_ignore_spans: bool,
    pub(super) body_resolver: BodyResolver<'a>,
    // Very often, we are hashing something that does not need the
    // `CachingSourceMapView`, so we initialize it lazily.
    raw_source_map: &'a SourceMap,
    caching_source_map: Option<CachingSourceMapView<'a>>,
    pub(super) hashing_controls: HashingControls,
}

/// The `BodyResolver` allows mapping a `BodyId` to the corresponding `hir::Body`.
/// We could also just store a plain reference to the `hir::Crate` but we want
/// to avoid that the crate is used to get untracked access to all of the HIR.
#[derive(Clone, Copy)]
pub(super) enum BodyResolver<'tcx> {
    Forbidden,
    Traverse {
        hash_bodies: bool,
        owner: LocalDefId,
        bodies: &'tcx SortedMap<hir::ItemLocalId, &'tcx hir::Body<'tcx>>,
    },
}

impl<'a> StableHashingContext<'a> {
    #[inline]
    fn new_with_or_without_spans(
        sess: &'a Session,
        definitions: &'a Definitions,
        cstore: &'a dyn CrateStore,
        always_ignore_spans: bool,
    ) -> Self {
        let hash_spans_initial =
            !always_ignore_spans && !sess.opts.debugging_opts.incremental_ignore_spans;

        StableHashingContext {
            body_resolver: BodyResolver::Forbidden,
            definitions,
            cstore,
            incremental_ignore_spans: sess.opts.debugging_opts.incremental_ignore_spans,
            caching_source_map: None,
            raw_source_map: sess.source_map(),
            hashing_controls: HashingControls {
                hash_spans: hash_spans_initial,
                node_id_hashing_mode: NodeIdHashingMode::HashDefPath,
            },
        }
    }

    #[inline]
    pub fn new(
        sess: &'a Session,
        definitions: &'a Definitions,
        cstore: &'a dyn CrateStore,
    ) -> Self {
        Self::new_with_or_without_spans(
            sess,
            definitions,
            cstore,
            /*always_ignore_spans=*/ false,
        )
    }

    #[inline]
    pub fn ignore_spans(
        sess: &'a Session,
        definitions: &'a Definitions,
        cstore: &'a dyn CrateStore,
    ) -> Self {
        let always_ignore_spans = true;
        Self::new_with_or_without_spans(sess, definitions, cstore, always_ignore_spans)
    }

    /// Allow hashing
    #[inline]
    pub fn while_hashing_hir_bodies(&mut self, hb: bool, f: impl FnOnce(&mut Self)) {
        let prev = match &mut self.body_resolver {
            BodyResolver::Forbidden => panic!("Hashing HIR bodies is forbidden."),
            BodyResolver::Traverse { ref mut hash_bodies, .. } => {
                std::mem::replace(hash_bodies, hb)
            }
        };
        f(self);
        match &mut self.body_resolver {
            BodyResolver::Forbidden => unreachable!(),
            BodyResolver::Traverse { ref mut hash_bodies, .. } => *hash_bodies = prev,
        }
    }

    #[inline]
    pub fn with_hir_bodies(
        &mut self,
        hash_bodies: bool,
        owner: LocalDefId,
        bodies: &'a SortedMap<hir::ItemLocalId, &'a hir::Body<'a>>,
        f: impl FnOnce(&mut Self),
    ) {
        let prev = self.body_resolver;
        self.body_resolver = BodyResolver::Traverse { hash_bodies, owner, bodies };
        f(self);
        self.body_resolver = prev;
    }

    #[inline]
    pub fn while_hashing_spans<F: FnOnce(&mut Self)>(&mut self, hash_spans: bool, f: F) {
        let prev_hash_spans = self.hashing_controls.hash_spans;
        self.hashing_controls.hash_spans = hash_spans;
        f(self);
        self.hashing_controls.hash_spans = prev_hash_spans;
    }

    #[inline]
    pub fn with_node_id_hashing_mode<F: FnOnce(&mut Self)>(
        &mut self,
        mode: NodeIdHashingMode,
        f: F,
    ) {
        let prev = self.hashing_controls.node_id_hashing_mode;
        self.hashing_controls.node_id_hashing_mode = mode;
        f(self);
        self.hashing_controls.node_id_hashing_mode = prev;
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

    #[inline]
    pub fn hashing_controls(&self) -> HashingControls {
        self.hashing_controls.clone()
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ast::NodeId {
    #[inline]
    fn hash_stable(&self, _: &mut StableHashingContext<'a>, _: &mut StableHasher) {
        panic!("Node IDs should not appear in incremental state");
    }
}

impl<'a> rustc_span::HashStableContext for StableHashingContext<'a> {
    #[inline]
    fn hash_spans(&self) -> bool {
        self.hashing_controls.hash_spans
    }

    #[inline]
    fn debug_opts_incremental_ignore_spans(&self) -> bool {
        self.incremental_ignore_spans
    }

    #[inline]
    fn def_path_hash(&self, def_id: DefId) -> DefPathHash {
        self.def_path_hash(def_id)
    }

    #[inline]
    fn def_span(&self, def_id: LocalDefId) -> Span {
        self.definitions.def_span(def_id)
    }

    #[inline]
    fn span_data_to_lines_and_cols(
        &mut self,
        span: &SpanData,
    ) -> Option<(Lrc<SourceFile>, usize, BytePos, usize, BytePos)> {
        self.source_map().span_data_to_lines_and_cols(span)
    }

    #[inline]
    fn hashing_controls(&self) -> HashingControls {
        self.hashing_controls.clone()
    }
}

impl<'a> rustc_session::HashStableContext for StableHashingContext<'a> {}
