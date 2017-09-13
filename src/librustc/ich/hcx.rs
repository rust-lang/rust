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
use ty::{self, TyCtxt, fast_reject};

use std::cmp::Ord;
use std::hash as std_hash;
use std::collections::HashMap;

use syntax::ast;
use syntax::attr;
use syntax::ext::hygiene::SyntaxContext;
use syntax::symbol::Symbol;
use syntax_pos::Span;

use rustc_data_structures::stable_hasher::{HashStable, StableHashingContextProvider,
                                           StableHasher, StableHasherResult,
                                           ToStableHashKey};
use rustc_data_structures::accumulate_vec::AccumulateVec;

/// This is the context state available during incr. comp. hashing. It contains
/// enough information to transform DefIds and HirIds into stable DefPaths (i.e.
/// a reference to the TyCtxt) and it holds a few caches for speeding up various
/// things (e.g. each DefId/DefPath is only hashed once).
pub struct StableHashingContext<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
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
}

impl<'a, 'gcx, 'tcx> StableHashingContext<'a, 'gcx, 'tcx> {

    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Self {
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
    pub fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> {
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

impl<'a, 'gcx, 'lcx> StableHashingContextProvider for ty::TyCtxt<'a, 'gcx, 'lcx> {
    type ContextType = StableHashingContext<'a, 'gcx, 'lcx>;
    fn create_stable_hashing_context(&self) -> Self::ContextType {
        StableHashingContext::new(*self)
    }
}

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::HirId {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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

                hcx.tcx.hir.definitions().def_path_hash(owner).hash_stable(hcx, hasher);
                local_id.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> ToStableHashKey<StableHashingContext<'a, 'gcx, 'tcx>> for hir::HirId {
    type KeyType = (DefPathHash, hir::ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self,
                          hcx: &StableHashingContext<'a, 'gcx, 'tcx>)
                          -> (DefPathHash, hir::ItemLocalId) {
        let def_path_hash = hcx.tcx().hir.definitions().def_path_hash(self.owner);
        (def_path_hash, self.local_id)
    }
}

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for ast::NodeId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        match hcx.node_id_hashing_mode {
            NodeIdHashingMode::Ignore => {
                // Don't do anything.
            }
            NodeIdHashingMode::HashDefPath => {
                hcx.tcx.hir.node_to_hir_id(*self).hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> ToStableHashKey<StableHashingContext<'a, 'gcx, 'tcx>> for ast::NodeId {
    type KeyType = (DefPathHash, hir::ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self,
                          hcx: &StableHashingContext<'a, 'gcx, 'tcx>)
                          -> (DefPathHash, hir::ItemLocalId) {
        hcx.tcx.hir.node_to_hir_id(*self).to_stable_hash_key(hcx)
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

pub fn hash_stable_trait_impls<'a, 'tcx, 'gcx, W, R>(
    hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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
        let tcx = hcx.tcx();
        let mut keys: AccumulateVec<[_; 8]> =
            non_blanket_impls.keys()
                             .map(|k| (k, k.map_def(|d| tcx.def_path_hash(d))))
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

