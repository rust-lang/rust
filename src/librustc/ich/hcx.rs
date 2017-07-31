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
use hir::def_id::DefId;
use hir::map::DefPathHash;
use ich::{self, CachingCodemapView};
use session::config::DebugInfoLevel::NoDebugInfo;
use ty;
use util::nodemap::{NodeMap, ItemLocalMap};

use std::hash as std_hash;
use std::collections::{HashMap, HashSet, BTreeMap};

use syntax::ast;
use syntax::attr;
use syntax::ext::hygiene::SyntaxContext;
use syntax::symbol::Symbol;
use syntax_pos::Span;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHasherResult};
use rustc_data_structures::accumulate_vec::AccumulateVec;

/// This is the context state available during incr. comp. hashing. It contains
/// enough information to transform DefIds and HirIds into stable DefPaths (i.e.
/// a reference to the TyCtxt) and it holds a few caches for speeding up various
/// things (e.g. each DefId/DefPath is only hashed once).
pub struct StableHashingContext<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: ty::TyCtxt<'a, 'gcx, 'tcx>,
    codemap: CachingCodemapView<'gcx>,
    hash_spans: bool,
    hash_bodies: bool,
    overflow_checks_enabled: bool,
    node_id_hashing_mode: NodeIdHashingMode,
    // A sorted array of symbol keys for fast lookup.
    ignored_attr_names: Vec<Symbol>,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum NodeIdHashingMode {
    Ignore,
    HashDefPath,
    HashTraitsInScope,
}

impl<'a, 'gcx, 'tcx> StableHashingContext<'a, 'gcx, 'tcx> {

    pub fn new(tcx: ty::TyCtxt<'a, 'gcx, 'tcx>) -> Self {
        let hash_spans_initial = tcx.sess.opts.debuginfo != NoDebugInfo;
        let check_overflow_initial = tcx.sess.overflow_checks();

        let mut ignored_attr_names: Vec<_> = ich::IGNORED_ATTRIBUTES
            .iter()
            .map(|&s| Symbol::intern(s))
            .collect();

        ignored_attr_names.sort();

        StableHashingContext {
            tcx,
            codemap: CachingCodemapView::new(tcx),
            hash_spans: hash_spans_initial,
            hash_bodies: true,
            overflow_checks_enabled: check_overflow_initial,
            node_id_hashing_mode: NodeIdHashingMode::HashDefPath,
            ignored_attr_names,
        }
    }

    pub fn force_span_hashing(mut self) -> Self {
        self.hash_spans = true;
        self
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
    pub fn tcx(&self) -> ty::TyCtxt<'a, 'gcx, 'tcx> {
        self.tcx
    }

    #[inline]
    pub fn def_path_hash(&mut self, def_id: DefId) -> DefPathHash {
        self.tcx.def_path_hash(def_id)
    }

    #[inline]
    pub fn hash_spans(&self) -> bool {
        self.hash_spans
    }

    #[inline]
    pub fn hash_bodies(&self) -> bool {
        self.hash_bodies
    }

    #[inline]
    pub fn codemap(&mut self) -> &mut CachingCodemapView<'gcx> {
        &mut self.codemap
    }

    #[inline]
    pub fn is_ignored_attr(&self, name: Symbol) -> bool {
        self.ignored_attr_names.binary_search(&name).is_ok()
    }

    pub fn hash_hir_item_like<F: FnOnce(&mut Self)>(&mut self,
                                                    item_attrs: &[ast::Attribute],
                                                    f: F) {
        let prev_overflow_checks = self.overflow_checks_enabled;
        if attr::contains_name(item_attrs, "rustc_inherit_overflow_checks") {
            self.overflow_checks_enabled = true;
        }
        let prev_hash_node_ids = self.node_id_hashing_mode;
        self.node_id_hashing_mode = NodeIdHashingMode::Ignore;

        f(self);

        self.node_id_hashing_mode = prev_hash_node_ids;
        self.overflow_checks_enabled = prev_overflow_checks;
    }

    #[inline]
    pub fn binop_can_panic_at_runtime(&self, binop: hir::BinOp_) -> bool
    {
        match binop {
            hir::BiAdd |
            hir::BiSub |
            hir::BiMul => self.overflow_checks_enabled,

            hir::BiDiv |
            hir::BiRem => true,

            hir::BiAnd |
            hir::BiOr |
            hir::BiBitXor |
            hir::BiBitAnd |
            hir::BiBitOr |
            hir::BiShl |
            hir::BiShr |
            hir::BiEq |
            hir::BiLt |
            hir::BiLe |
            hir::BiNe |
            hir::BiGe |
            hir::BiGt => false
        }
    }

    #[inline]
    pub fn unop_can_panic_at_runtime(&self, unop: hir::UnOp) -> bool
    {
        match unop {
            hir::UnDeref |
            hir::UnNot => false,
            hir::UnNeg => self.overflow_checks_enabled,
        }
    }
}


impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for ast::NodeId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        match hcx.node_id_hashing_mode {
            NodeIdHashingMode::Ignore => {
                // Most NodeIds in the HIR can be ignored, but if there is a
                // corresponding entry in the `trait_map` we need to hash that.
                // Make sure we don't ignore too much by checking that there is
                // no entry in a debug_assert!().
                debug_assert!(hcx.tcx.trait_map.get(self).is_none());
            }
            NodeIdHashingMode::HashDefPath => {
                hcx.tcx.hir.definitions().node_to_hir_id(*self).hash_stable(hcx, hasher);
            }
            NodeIdHashingMode::HashTraitsInScope => {
                if let Some(traits) = hcx.tcx.trait_map.get(self) {
                    // The ordering of the candidates is not fixed. So we hash
                    // the def-ids and then sort them and hash the collection.
                    let mut candidates: AccumulateVec<[_; 8]> =
                        traits.iter()
                              .map(|&hir::TraitCandidate { def_id, import_id: _ }| {
                                  hcx.def_path_hash(def_id)
                              })
                              .collect();
                    if traits.len() > 1 {
                        candidates.sort();
                    }
                    candidates.hash_stable(hcx, hasher);
                }
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for Span {

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
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        use syntax_pos::Pos;

        if !hcx.hash_spans {
            return
        }

        // If this is not an empty or invalid span, we want to hash the last
        // position that belongs to it, as opposed to hashing the first
        // position past it.
        let span_hi = if self.hi() > self.lo() {
            // We might end up in the middle of a multibyte character here,
            // but that's OK, since we are not trying to decode anything at
            // this position.
            self.hi() - ::syntax_pos::BytePos(1)
        } else {
            self.hi()
        };

        {
            let loc1 = hcx.codemap().byte_pos_to_line_and_col(self.lo());
            let loc1 = loc1.as_ref()
                           .map(|&(ref fm, line, col)| (&fm.name[..], line, col.to_usize()))
                           .unwrap_or(("???", 0, 0));

            let loc2 = hcx.codemap().byte_pos_to_line_and_col(span_hi);
            let loc2 = loc2.as_ref()
                           .map(|&(ref fm, line, col)| (&fm.name[..], line, col.to_usize()))
                           .unwrap_or(("???", 0, 0));

            if loc1.0 == loc2.0 {
                std_hash::Hash::hash(&0u8, hasher);

                std_hash::Hash::hash(loc1.0, hasher);
                std_hash::Hash::hash(&loc1.1, hasher);
                std_hash::Hash::hash(&loc1.2, hasher);

                // Do not hash the file name twice
                std_hash::Hash::hash(&loc2.1, hasher);
                std_hash::Hash::hash(&loc2.2, hasher);
            } else {
                std_hash::Hash::hash(&1u8, hasher);

                std_hash::Hash::hash(loc1.0, hasher);
                std_hash::Hash::hash(&loc1.1, hasher);
                std_hash::Hash::hash(&loc1.2, hasher);

                std_hash::Hash::hash(loc2.0, hasher);
                std_hash::Hash::hash(&loc2.1, hasher);
                std_hash::Hash::hash(&loc2.2, hasher);
            }
        }

        if self.ctxt() == SyntaxContext::empty() {
            0u8.hash_stable(hcx, hasher);
        } else {
            1u8.hash_stable(hcx, hasher);
            self.source_callsite().hash_stable(hcx, hasher);
        }
    }
}

pub fn hash_stable_hashmap<'a, 'gcx, 'tcx, K, V, R, SK, F, W>(
    hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
    hasher: &mut StableHasher<W>,
    map: &HashMap<K, V, R>,
    extract_stable_key: F)
    where K: Eq + std_hash::Hash,
          V: HashStable<StableHashingContext<'a, 'gcx, 'tcx>>,
          R: std_hash::BuildHasher,
          SK: HashStable<StableHashingContext<'a, 'gcx, 'tcx>> + Ord + Clone,
          F: Fn(&mut StableHashingContext<'a, 'gcx, 'tcx>, &K) -> SK,
          W: StableHasherResult,
{
    let mut keys: Vec<_> = map.keys()
                              .map(|k| (extract_stable_key(hcx, k), k))
                              .collect();
    keys.sort_unstable_by_key(|&(ref stable_key, _)| stable_key.clone());
    keys.len().hash_stable(hcx, hasher);
    for (stable_key, key) in keys {
        stable_key.hash_stable(hcx, hasher);
        map[key].hash_stable(hcx, hasher);
    }
}

pub fn hash_stable_hashset<'a, 'tcx, 'gcx, K, R, SK, F, W>(
    hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
    hasher: &mut StableHasher<W>,
    set: &HashSet<K, R>,
    extract_stable_key: F)
    where K: Eq + std_hash::Hash,
          R: std_hash::BuildHasher,
          SK: HashStable<StableHashingContext<'a, 'gcx, 'tcx>> + Ord + Clone,
          F: Fn(&mut StableHashingContext<'a, 'gcx, 'tcx>, &K) -> SK,
          W: StableHasherResult,
{
    let mut keys: Vec<_> = set.iter()
                              .map(|k| extract_stable_key(hcx, k))
                              .collect();
    keys.sort_unstable();
    keys.hash_stable(hcx, hasher);
}

pub fn hash_stable_nodemap<'a, 'tcx, 'gcx, V, W>(
    hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
    hasher: &mut StableHasher<W>,
    map: &NodeMap<V>)
    where V: HashStable<StableHashingContext<'a, 'gcx, 'tcx>>,
          W: StableHasherResult,
{
    hash_stable_hashmap(hcx, hasher, map, |hcx, node_id| {
        hcx.tcx.hir.definitions().node_to_hir_id(*node_id).local_id
    });
}

pub fn hash_stable_itemlocalmap<'a, 'tcx, 'gcx, V, W>(
    hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
    hasher: &mut StableHasher<W>,
    map: &ItemLocalMap<V>)
    where V: HashStable<StableHashingContext<'a, 'gcx, 'tcx>>,
          W: StableHasherResult,
{
    hash_stable_hashmap(hcx, hasher, map, |_, local_id| {
        *local_id
    });
}


pub fn hash_stable_btreemap<'a, 'tcx, 'gcx, K, V, SK, F, W>(
    hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
    hasher: &mut StableHasher<W>,
    map: &BTreeMap<K, V>,
    extract_stable_key: F)
    where K: Eq + Ord,
          V: HashStable<StableHashingContext<'a, 'gcx, 'tcx>>,
          SK: HashStable<StableHashingContext<'a, 'gcx, 'tcx>> + Ord + Clone,
          F: Fn(&mut StableHashingContext<'a, 'gcx, 'tcx>, &K) -> SK,
          W: StableHasherResult,
{
    let mut keys: Vec<_> = map.keys()
                              .map(|k| (extract_stable_key(hcx, k), k))
                              .collect();
    keys.sort_unstable_by_key(|&(ref stable_key, _)| stable_key.clone());
    keys.len().hash_stable(hcx, hasher);
    for (stable_key, key) in keys {
        stable_key.hash_stable(hcx, hasher);
        map[key].hash_stable(hcx, hasher);
    }
}
