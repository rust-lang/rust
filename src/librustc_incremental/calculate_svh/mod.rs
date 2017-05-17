// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Calculation of the (misnamed) "strict version hash" for crates and
//! items. This hash is used to tell when the HIR changed in such a
//! way that results from previous compilations may no longer be
//! applicable and hence must be recomputed. It should probably be
//! renamed to the ICH (incremental compilation hash).
//!
//! The hashes for all items are computed once at the beginning of
//! compilation and stored into a map. In addition, a hash is computed
//! of the **entire crate**.
//!
//! Storing the hashes in a map avoids the need to compute them twice
//! (once when loading prior incremental results and once when
//! saving), but it is also important for correctness: at least as of
//! the time of this writing, the typeck passes rewrites entries in
//! the dep-map in-place to accommodate UFCS resolutions. Since name
//! resolution is part of the hash, the result is that hashes computed
//! at the end of compilation would be different from those computed
//! at the beginning.

use std::cell::RefCell;
use std::hash::Hash;
use std::sync::Arc;
use rustc::dep_graph::DepNode;
use rustc::hir;
use rustc::hir::def_id::{LOCAL_CRATE, CRATE_DEF_INDEX, DefId};
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::ich::{Fingerprint, StableHashingContext};
use rustc::ty::TyCtxt;
use rustc::util::common::record_time;
use rustc_data_structures::stable_hasher::{StableHasher, HashStable};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::accumulate_vec::AccumulateVec;

pub type IchHasher = StableHasher<Fingerprint>;

pub struct IncrementalHashesMap {
    hashes: FxHashMap<DepNode<DefId>, Fingerprint>,

    // These are the metadata hashes for the current crate as they were stored
    // during the last compilation session. They are only loaded if
    // -Z query-dep-graph was specified and are needed for auto-tests using
    // the #[rustc_metadata_dirty] and #[rustc_metadata_clean] attributes to
    // check whether some metadata hash has changed in between two revisions.
    pub prev_metadata_hashes: RefCell<FxHashMap<DefId, Fingerprint>>,
}

impl IncrementalHashesMap {
    pub fn new() -> IncrementalHashesMap {
        IncrementalHashesMap {
            hashes: FxHashMap(),
            prev_metadata_hashes: RefCell::new(FxHashMap()),
        }
    }

    pub fn get(&self, k: &DepNode<DefId>) -> Option<&Fingerprint> {
        self.hashes.get(k)
    }

    pub fn insert(&mut self, k: DepNode<DefId>, v: Fingerprint) -> Option<Fingerprint> {
        self.hashes.insert(k, v)
    }

    pub fn iter<'a>(&'a self)
                    -> ::std::collections::hash_map::Iter<'a, DepNode<DefId>, Fingerprint> {
        self.hashes.iter()
    }

    pub fn len(&self) -> usize {
        self.hashes.len()
    }
}

impl<'a> ::std::ops::Index<&'a DepNode<DefId>> for IncrementalHashesMap {
    type Output = Fingerprint;

    fn index(&self, index: &'a DepNode<DefId>) -> &Fingerprint {
        match self.hashes.get(index) {
            Some(fingerprint) => fingerprint,
            None => {
                bug!("Could not find ICH for {:?}", index);
            }
        }
    }
}

struct ComputeItemHashesVisitor<'a, 'tcx: 'a> {
    hcx: StableHashingContext<'a, 'tcx>,
    hashes: IncrementalHashesMap,
}

impl<'a, 'tcx: 'a> ComputeItemHashesVisitor<'a, 'tcx> {
    fn compute_and_store_ich_for_item_like<T>(&mut self,
                                              dep_node: DepNode<DefId>,
                                              hash_bodies: bool,
                                              item_like: T)
        where T: HashStable<StableHashingContext<'a, 'tcx>>
    {
        if !hash_bodies && !self.hcx.tcx().sess.opts.build_dep_graph() {
            // If we just need the hashes in order to compute the SVH, we don't
            // need have two hashes per item. Just the one containing also the
            // item's body is sufficient.
            return
        }

        let mut hasher = IchHasher::new();
        self.hcx.while_hashing_hir_bodies(hash_bodies, |hcx| {
            item_like.hash_stable(hcx, &mut hasher);
        });

        let bytes_hashed = hasher.bytes_hashed();
        let item_hash = hasher.finish();
        debug!("calculate_def_hash: dep_node={:?} hash={:?}", dep_node, item_hash);
        self.hashes.insert(dep_node, item_hash);

        let tcx = self.hcx.tcx();
        let bytes_hashed =
            tcx.sess.perf_stats.incr_comp_bytes_hashed.get() +
            bytes_hashed;
        tcx.sess.perf_stats.incr_comp_bytes_hashed.set(bytes_hashed);
    }

    fn compute_crate_hash(&mut self) {
        let tcx = self.hcx.tcx();
        let krate = tcx.hir.krate();

        let mut crate_state = IchHasher::new();

        let crate_disambiguator = tcx.sess.local_crate_disambiguator();
        "crate_disambiguator".hash(&mut crate_state);
        crate_disambiguator.as_str().len().hash(&mut crate_state);
        crate_disambiguator.as_str().hash(&mut crate_state);

        // add each item (in some deterministic order) to the overall
        // crate hash.
        {
            let hcx = &mut self.hcx;
            let mut item_hashes: Vec<_> =
                self.hashes.iter()
                           .filter_map(|(item_dep_node, &item_hash)| {
                                // This `match` determines what kinds of nodes
                                // go into the SVH:
                                match *item_dep_node {
                                    DepNode::Hir(_) |
                                    DepNode::HirBody(_) => {
                                        // We want to incoporate these into the
                                        // SVH.
                                    }
                                    DepNode::FileMap(..) => {
                                        // These don't make a semantic
                                        // difference, filter them out.
                                        return None
                                    }
                                    DepNode::AllLocalTraitImpls => {
                                        // These are already covered by hashing
                                        // the HIR.
                                        return None
                                    }
                                    ref other => {
                                        bug!("Found unexpected DepNode during \
                                              SVH computation: {:?}",
                                             other)
                                    }
                                }

                                // Convert from a DepNode<DefId> to a
                                // DepNode<u64> where the u64 is the hash of
                                // the def-id's def-path:
                                let item_dep_node =
                                    item_dep_node.map_def(|&did| Some(hcx.def_path_hash(did)))
                                                 .unwrap();
                                Some((item_dep_node, item_hash))
                           })
                           .collect();
            item_hashes.sort_unstable(); // avoid artificial dependencies on item ordering
            item_hashes.hash(&mut crate_state);
        }

        krate.attrs.hash_stable(&mut self.hcx, &mut crate_state);

        let crate_hash = crate_state.finish();
        self.hashes.insert(DepNode::Krate, crate_hash);
        debug!("calculate_crate_hash: crate_hash={:?}", crate_hash);
    }

    fn hash_crate_root_module(&mut self, krate: &'tcx hir::Crate) {
        let hir::Crate {
            ref module,
            // Crate attributes are not copied over to the root `Mod`, so hash
            // them explicitly here.
            ref attrs,
            span,

            // These fields are handled separately:
            exported_macros: _,
            items: _,
            trait_items: _,
            impl_items: _,
            bodies: _,
            trait_impls: _,
            trait_default_impl: _,
            body_ids: _,
        } = *krate;

        let def_id = DefId::local(CRATE_DEF_INDEX);
        self.compute_and_store_ich_for_item_like(DepNode::Hir(def_id),
                                                 false,
                                                 (module, (span, attrs)));
        self.compute_and_store_ich_for_item_like(DepNode::HirBody(def_id),
                                                 true,
                                                 (module, (span, attrs)));
    }

    fn compute_and_store_ich_for_trait_impls(&mut self, krate: &'tcx hir::Crate)
    {
        let tcx = self.hcx.tcx();

        let mut impls: Vec<(u64, Fingerprint)> = krate
            .trait_impls
            .iter()
            .map(|(&trait_id, impls)| {
                let trait_id = tcx.def_path_hash(trait_id);
                let mut impls: AccumulateVec<[_; 32]> = impls
                    .iter()
                    .map(|&node_id| {
                        let def_id = tcx.hir.local_def_id(node_id);
                        tcx.def_path_hash(def_id)
                    })
                    .collect();

                impls.sort_unstable();
                let mut hasher = StableHasher::new();
                impls.hash_stable(&mut self.hcx, &mut hasher);
                (trait_id, hasher.finish())
            })
            .collect();

        impls.sort_unstable();

        let mut default_impls: AccumulateVec<[_; 32]> = krate
            .trait_default_impl
            .iter()
            .map(|(&trait_def_id, &impl_node_id)| {
                let impl_def_id = tcx.hir.local_def_id(impl_node_id);
                (tcx.def_path_hash(trait_def_id), tcx.def_path_hash(impl_def_id))
            })
            .collect();

        default_impls.sort_unstable();

        let mut hasher = StableHasher::new();
        impls.hash_stable(&mut self.hcx, &mut hasher);

        self.hashes.insert(DepNode::AllLocalTraitImpls, hasher.finish());
    }
}

impl<'a, 'tcx: 'a> ItemLikeVisitor<'tcx> for ComputeItemHashesVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let def_id = self.hcx.tcx().hir.local_def_id(item.id);
        self.compute_and_store_ich_for_item_like(DepNode::Hir(def_id), false, item);
        self.compute_and_store_ich_for_item_like(DepNode::HirBody(def_id), true, item);
    }

    fn visit_trait_item(&mut self, item: &'tcx hir::TraitItem) {
        let def_id = self.hcx.tcx().hir.local_def_id(item.id);
        self.compute_and_store_ich_for_item_like(DepNode::Hir(def_id), false, item);
        self.compute_and_store_ich_for_item_like(DepNode::HirBody(def_id), true, item);
    }

    fn visit_impl_item(&mut self, item: &'tcx hir::ImplItem) {
        let def_id = self.hcx.tcx().hir.local_def_id(item.id);
        self.compute_and_store_ich_for_item_like(DepNode::Hir(def_id), false, item);
        self.compute_and_store_ich_for_item_like(DepNode::HirBody(def_id), true, item);
    }
}



pub fn compute_incremental_hashes_map<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                                    -> IncrementalHashesMap {
    let _ignore = tcx.dep_graph.in_ignore();
    let krate = tcx.hir.krate();

    let mut visitor = ComputeItemHashesVisitor {
        hcx: StableHashingContext::new(tcx),
        hashes: IncrementalHashesMap::new(),
    };

    record_time(&tcx.sess.perf_stats.incr_comp_hashes_time, || {
        visitor.hash_crate_root_module(krate);
        krate.visit_all_item_likes(&mut visitor);

        for macro_def in krate.exported_macros.iter() {
            let def_id = tcx.hir.local_def_id(macro_def.id);
            visitor.compute_and_store_ich_for_item_like(DepNode::Hir(def_id), false, macro_def);
            visitor.compute_and_store_ich_for_item_like(DepNode::HirBody(def_id), true, macro_def);
        }

        for filemap in tcx.sess
                          .codemap()
                          .files_untracked()
                          .iter()
                          .filter(|fm| !fm.is_imported()) {
            assert_eq!(LOCAL_CRATE.as_u32(), filemap.crate_of_origin);
            let def_id = DefId {
                krate: LOCAL_CRATE,
                index: CRATE_DEF_INDEX,
            };
            let name = Arc::new(filemap.name.clone());
            let dep_node = DepNode::FileMap(def_id, name);
            let mut hasher = IchHasher::new();
            filemap.hash_stable(&mut visitor.hcx, &mut hasher);
            let fingerprint = hasher.finish();
            visitor.hashes.insert(dep_node, fingerprint);
        }

        visitor.compute_and_store_ich_for_trait_impls(krate);
    });

    tcx.sess.perf_stats.incr_comp_hashes_count.set(visitor.hashes.len() as u64);

    record_time(&tcx.sess.perf_stats.svh_time, || visitor.compute_crate_hash());
    visitor.hashes
}
