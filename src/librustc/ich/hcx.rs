use crate::hir;
use crate::hir::def_id::{DefId, DefIndex};
use crate::hir::map::DefPathHash;
use crate::hir::map::definitions::Definitions;
use crate::ich::{self, CachingSourceMapView, Fingerprint};
use crate::middle::cstore::CrateStore;
use crate::ty::{TyCtxt, fast_reject};
use crate::session::Session;

use std::cmp::Ord;
use std::hash as std_hash;
use std::cell::RefCell;

use syntax::ast;

use syntax::source_map::SourceMap;
use syntax::ext::hygiene::SyntaxContext;
use syntax::symbol::Symbol;
use syntax::tokenstream::DelimSpan;
use syntax_pos::{Span, DUMMY_SP};
use syntax_pos::hygiene;

use rustc_data_structures::stable_hasher::{HashStable,
                                           StableHasher, StableHasherResult,
                                           ToStableHashKey};
use rustc_data_structures::fx::{FxHashSet, FxHashMap};
use smallvec::SmallVec;

fn compute_ignored_attr_names() -> FxHashSet<Symbol> {
    debug_assert!(ich::IGNORED_ATTRIBUTES.len() > 0);
    ich::IGNORED_ATTRIBUTES.iter().map(|&s| s).collect()
}

/// This is the context state available during incr. comp. hashing. It contains
/// enough information to transform DefIds and HirIds into stable DefPaths (i.e.
/// a reference to the TyCtxt) and it holds a few caches for speeding up various
/// things (e.g., each DefId/DefPath is only hashed once).
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
    // CachingSourceMapView, so we initialize it lazily.
    raw_source_map: &'a SourceMap,
    caching_source_map: Option<CachingSourceMapView<'a>>,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum NodeIdHashingMode {
    Ignore,
    HashDefPath,
}

/// The BodyResolver allows to map a BodyId to the corresponding hir::Body.
/// We could also just store a plain reference to the hir::Crate but we want
/// to avoid that the crate is used to get untracked access to all of the HIR.
#[derive(Clone, Copy)]
struct BodyResolver<'tcx>(&'tcx hir::Crate);

impl<'tcx> BodyResolver<'tcx> {
    // Return a reference to the hir::Body with the given BodyId.
    // DOES NOT DO ANY TRACKING, use carefully.
    fn body(self, id: hir::BodyId) -> &'tcx hir::Body {
        self.0.body(id)
    }
}

impl<'a> StableHashingContext<'a> {
    // The `krate` here is only used for mapping BodyIds to Bodies.
    // Don't use it for anything else or you'll run the risk of
    // leaking data out of the tracking system.
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

impl<'a> crate::dep_graph::DepGraphSafe for StableHashingContext<'a> {
}


impl<'a> HashStable<StableHashingContext<'a>> for hir::BodyId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        if hcx.hash_bodies() {
            hcx.body_resolver.body(*self).hash_stable(hcx, hasher);
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::HirId {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
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
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
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

impl<'a> HashStable<StableHashingContext<'a>> for Span {

    // Hash a span in a stable way. We can't directly hash the span's BytePos
    // fields (that would be similar to hashing pointers, since those are just
    // offsets into the SourceMap). Instead, we hash the (file name, line, column)
    // triple, which stays the same even if the containing SourceFile has moved
    // within the SourceMap.
    // Also note that we are hashing byte offsets for the column, not unicode
    // codepoint offsets. For the purpose of the hash that's sufficient.
    // Also, hashing filenames is expensive so we avoid doing it twice when the
    // span starts and ends in the same file, which is almost always the case.
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        const TAG_VALID_SPAN: u8 = 0;
        const TAG_INVALID_SPAN: u8 = 1;
        const TAG_EXPANSION: u8 = 0;
        const TAG_NO_EXPANSION: u8 = 1;

        if !hcx.hash_spans {
            return
        }

        if *self == DUMMY_SP {
            return std_hash::Hash::hash(&TAG_INVALID_SPAN, hasher);
        }

        // If this is not an empty or invalid span, we want to hash the last
        // position that belongs to it, as opposed to hashing the first
        // position past it.
        let span = self.data();

        if span.hi < span.lo {
            return std_hash::Hash::hash(&TAG_INVALID_SPAN, hasher);
        }

        let (file_lo, line_lo, col_lo) = match hcx.source_map()
                                                  .byte_pos_to_line_and_col(span.lo) {
            Some(pos) => pos,
            None => {
                return std_hash::Hash::hash(&TAG_INVALID_SPAN, hasher);
            }
        };

        if !file_lo.contains(span.hi) {
            return std_hash::Hash::hash(&TAG_INVALID_SPAN, hasher);
        }

        std_hash::Hash::hash(&TAG_VALID_SPAN, hasher);
        // We truncate the stable_id hash and line and col numbers. The chances
        // of causing a collision this way should be minimal.
        std_hash::Hash::hash(&(file_lo.name_hash as u64), hasher);

        let col = (col_lo.0 as u64) & 0xFF;
        let line = ((line_lo as u64) & 0xFF_FF_FF) << 8;
        let len = ((span.hi - span.lo).0 as u64) << 32;
        let line_col_len = col | line | len;
        std_hash::Hash::hash(&line_col_len, hasher);

        if span.ctxt == SyntaxContext::empty() {
            TAG_NO_EXPANSION.hash_stable(hcx, hasher);
        } else {
            TAG_EXPANSION.hash_stable(hcx, hasher);

            // Since the same expansion context is usually referenced many
            // times, we cache a stable hash of it and hash that instead of
            // recursing every time.
            thread_local! {
                static CACHE: RefCell<FxHashMap<hygiene::Mark, u64>> = Default::default();
            }

            let sub_hash: u64 = CACHE.with(|cache| {
                let mark = span.ctxt.outer();

                if let Some(&sub_hash) = cache.borrow().get(&mark) {
                    return sub_hash;
                }

                let mut hasher = StableHasher::new();
                mark.expn_info().hash_stable(hcx, &mut hasher);
                let sub_hash: Fingerprint = hasher.finish();
                let sub_hash = sub_hash.to_smaller_hash();
                cache.borrow_mut().insert(mark, sub_hash);
                sub_hash
            });

            sub_hash.hash_stable(hcx, hasher);
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for DelimSpan {
    fn hash_stable<W: StableHasherResult>(
        &self,
        hcx: &mut StableHashingContext<'a>,
        hasher: &mut StableHasher<W>,
    ) {
        self.open.hash_stable(hcx, hasher);
        self.close.hash_stable(hcx, hasher);
    }
}

pub fn hash_stable_trait_impls<'a, W>(
    hcx: &mut StableHashingContext<'a>,
    hasher: &mut StableHasher<W>,
    blanket_impls: &[DefId],
    non_blanket_impls: &FxHashMap<fast_reject::SimplifiedType, Vec<DefId>>,
) where
    W: StableHasherResult,
{
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
