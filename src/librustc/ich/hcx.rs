// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir;
use hir::def_id::{DefId, DefIndex};
use hir::map::DefPathHash;
use hir::map::definitions::Definitions;
use ich::{self, CachingCodemapView};
use middle::cstore::CrateStore;
use ty::{TyCtxt, fast_reject};
use session::Session;

use std::cmp::Ord;
use std::hash as std_hash;
use std::cell::RefCell;
use std::collections::HashMap;

use syntax::ast;

use syntax::codemap::CodeMap;
use syntax::ext::hygiene::SyntaxContext;
use syntax::symbol::Symbol;
use syntax_pos::{Span, DUMMY_SP};

use rustc_data_structures::stable_hasher::{HashStable, StableHashingContextProvider,
                                           StableHasher, StableHasherResult,
                                           ToStableHashKey};
use rustc_data_structures::accumulate_vec::AccumulateVec;
use rustc_data_structures::fx::FxHashSet;

thread_local!(static IGNORED_ATTR_NAMES: RefCell<FxHashSet<Symbol>> =
    RefCell::new(FxHashSet()));

/// This is the context state available during incr. comp. hashing. It contains
/// enough information to transform DefIds and HirIds into stable DefPaths (i.e.
/// a reference to the TyCtxt) and it holds a few caches for speeding up various
/// things (e.g. each DefId/DefPath is only hashed once).
#[derive(Clone)]
pub struct StableHashingContext<'gcx> {
    sess: &'gcx Session,
    definitions: &'gcx Definitions,
    cstore: &'gcx CrateStore,
    body_resolver: BodyResolver<'gcx>,
    hash_spans: bool,
    hash_bodies: bool,
    node_id_hashing_mode: NodeIdHashingMode,

    // Very often, we are hashing something that does not need the
    // CachingCodemapView, so we initialize it lazily.
    raw_codemap: &'gcx CodeMap,
    caching_codemap: Option<CachingCodemapView<'gcx>>,
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
struct BodyResolver<'gcx>(&'gcx hir::Crate);

impl<'gcx> BodyResolver<'gcx> {
    // Return a reference to the hir::Body with the given BodyId.
    // DOES NOT DO ANY TRACKING, use carefully.
    fn body(self, id: hir::BodyId) -> &'gcx hir::Body {
        self.0.body(id)
    }
}

impl<'gcx> StableHashingContext<'gcx> {
    // The `krate` here is only used for mapping BodyIds to Bodies.
    // Don't use it for anything else or you'll run the risk of
    // leaking data out of the tracking system.
    pub fn new(sess: &'gcx Session,
               krate: &'gcx hir::Crate,
               definitions: &'gcx Definitions,
               cstore: &'gcx CrateStore)
               -> Self {
        let hash_spans_initial = !sess.opts.debugging_opts.incremental_ignore_spans;

        debug_assert!(ich::IGNORED_ATTRIBUTES.len() > 0);
        IGNORED_ATTR_NAMES.with(|names| {
            let mut names = names.borrow_mut();
            if names.is_empty() {
                names.extend(ich::IGNORED_ATTRIBUTES.iter()
                                                    .map(|&s| Symbol::intern(s)));
            }
        });

        StableHashingContext {
            sess,
            body_resolver: BodyResolver(krate),
            definitions,
            cstore,
            caching_codemap: None,
            raw_codemap: sess.codemap(),
            hash_spans: hash_spans_initial,
            hash_bodies: true,
            node_id_hashing_mode: NodeIdHashingMode::HashDefPath,
        }
    }

    #[inline]
    pub fn sess(&self) -> &'gcx Session {
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
    pub fn codemap(&mut self) -> &mut CachingCodemapView<'gcx> {
        match self.caching_codemap {
            Some(ref mut cm) => {
                cm
            }
            ref mut none => {
                *none = Some(CachingCodemapView::new(self.raw_codemap));
                none.as_mut().unwrap()
            }
        }
    }

    #[inline]
    pub fn is_ignored_attr(&self, name: Symbol) -> bool {
        IGNORED_ATTR_NAMES.with(|names| {
            names.borrow().contains(&name)
        })
    }

    pub fn hash_hir_item_like<F: FnOnce(&mut Self)>(&mut self, f: F) {
        let prev_hash_node_ids = self.node_id_hashing_mode;
        self.node_id_hashing_mode = NodeIdHashingMode::Ignore;

        f(self);

        self.node_id_hashing_mode = prev_hash_node_ids;
    }
}

impl<'a, 'gcx, 'lcx> StableHashingContextProvider for TyCtxt<'a, 'gcx, 'lcx> {
    type ContextType = StableHashingContext<'gcx>;
    fn create_stable_hashing_context(&self) -> Self::ContextType {
        (*self).create_stable_hashing_context()
    }
}


impl<'gcx> StableHashingContextProvider for StableHashingContext<'gcx> {
    type ContextType = StableHashingContext<'gcx>;
    fn create_stable_hashing_context(&self) -> Self::ContextType {
        self.clone()
    }
}

impl<'gcx> ::dep_graph::DepGraphSafe for StableHashingContext<'gcx> {
}


impl<'gcx> HashStable<StableHashingContext<'gcx>> for hir::BodyId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        if hcx.hash_bodies() {
            hcx.body_resolver.body(*self).hash_stable(hcx, hasher);
        }
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for hir::HirId {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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

impl<'gcx> ToStableHashKey<StableHashingContext<'gcx>> for hir::HirId {
    type KeyType = (DefPathHash, hir::ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self,
                          hcx: &StableHashingContext<'gcx>)
                          -> (DefPathHash, hir::ItemLocalId) {
        let def_path_hash = hcx.local_def_path_hash(self.owner);
        (def_path_hash, self.local_id)
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for ast::NodeId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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

impl<'gcx> ToStableHashKey<StableHashingContext<'gcx>> for ast::NodeId {
    type KeyType = (DefPathHash, hir::ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self,
                          hcx: &StableHashingContext<'gcx>)
                          -> (DefPathHash, hir::ItemLocalId) {
        hcx.definitions.node_to_hir_id(*self).to_stable_hash_key(hcx)
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for Span {

    // Hash a span in a stable way. We can't directly hash the span's BytePos
    // fields (that would be similar to hashing pointers, since those are just
    // offsets into the CodeMap). Instead, we hash the (file name, line, column)
    // triple, which stays the same even if the containing FileMap has moved
    // within the CodeMap.
    // Also note that we are hashing byte offsets for the column, not unicode
    // codepoint offsets. For the purpose of the hash that's sufficient.
    // Also, hashing filenames is expensive so we avoid doing it twice when the
    // span starts and ends in the same file, which is almost always the case.
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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

        let (file_lo, line_lo, col_lo) = match hcx.codemap()
                                                  .byte_pos_to_line_and_col(span.lo) {
            Some(pos) => pos,
            None => {
                return std_hash::Hash::hash(&TAG_INVALID_SPAN, hasher);
            }
        };

        if !file_lo.contains(span.hi) {
            return std_hash::Hash::hash(&TAG_INVALID_SPAN, hasher);
        }

        let len = span.hi - span.lo;

        std_hash::Hash::hash(&TAG_VALID_SPAN, hasher);
        std_hash::Hash::hash(&file_lo.name, hasher);
        std_hash::Hash::hash(&line_lo, hasher);
        std_hash::Hash::hash(&col_lo, hasher);
        std_hash::Hash::hash(&len, hasher);

        if span.ctxt == SyntaxContext::empty() {
            TAG_NO_EXPANSION.hash_stable(hcx, hasher);
        } else {
            TAG_EXPANSION.hash_stable(hcx, hasher);
            span.ctxt.outer().expn_info().hash_stable(hcx, hasher);
        }
    }
}

pub fn hash_stable_trait_impls<'gcx, W, R>(
    hcx: &mut StableHashingContext<'gcx>,
    hasher: &mut StableHasher<W>,
    blanket_impls: &Vec<DefId>,
    non_blanket_impls: &HashMap<fast_reject::SimplifiedType, Vec<DefId>, R>)
    where W: StableHasherResult,
          R: std_hash::BuildHasher,
{
    {
        let mut blanket_impls: AccumulateVec<[_; 8]> = blanket_impls
            .iter()
            .map(|&def_id| hcx.def_path_hash(def_id))
            .collect();

        if blanket_impls.len() > 1 {
            blanket_impls.sort_unstable();
        }

        blanket_impls.hash_stable(hcx, hasher);
    }

    {
        let mut keys: AccumulateVec<[_; 8]> =
            non_blanket_impls.keys()
                             .map(|k| (k, k.map_def(|d| hcx.def_path_hash(d))))
                             .collect();
        keys.sort_unstable_by(|&(_, ref k1), &(_, ref k2)| k1.cmp(k2));
        keys.len().hash_stable(hcx, hasher);
        for (key, ref stable_key) in keys {
            stable_key.hash_stable(hcx, hasher);
            let mut impls : AccumulateVec<[_; 8]> = non_blanket_impls[key]
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

