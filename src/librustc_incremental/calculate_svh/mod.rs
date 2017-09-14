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
use rustc::dep_graph::{DepNode, DepKind};
use rustc::hir;
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId, DefIndex};
use rustc::hir::map::DefPathHash;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::ich::{Fingerprint, StableHashingContext};
use rustc::ty::TyCtxt;
use rustc::util::common::record_time;
use rustc_data_structures::stable_hasher::{StableHasher, HashStable};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::accumulate_vec::AccumulateVec;

pub type IchHasher = StableHasher<Fingerprint>;

pub struct IncrementalHashesMap {
    hashes: FxHashMap<DepNode, Fingerprint>,

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

    pub fn insert(&mut self, k: DepNode, v: Fingerprint) {
        assert!(self.hashes.insert(k, v).is_none());
    }

    pub fn iter<'a>(&'a self)
                    -> ::std::collections::hash_map::Iter<'a, DepNode, Fingerprint> {
        self.hashes.iter()
    }

    pub fn len(&self) -> usize {
        self.hashes.len()
    }
}

impl<'a> ::std::ops::Index<&'a DepNode> for IncrementalHashesMap {
    type Output = Fingerprint;

    fn index(&self, index: &'a DepNode) -> &Fingerprint {
        match self.hashes.get(index) {
            Some(fingerprint) => fingerprint,
            None => {
                bug!("Could not find ICH for {:?}", index);
            }
        }
    }
}

struct ComputeItemHashesVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    hcx: StableHashingContext<'tcx>,
    hashes: IncrementalHashesMap,
}

impl<'a, 'tcx: 'a> ComputeItemHashesVisitor<'a, 'tcx> {
    fn compute_and_store_ich_for_item_like<T>(&mut self,
                                              def_index: DefIndex,
                                              hash_bodies: bool,
                                              item_like: T)
        where T: HashStable<StableHashingContext<'tcx>>
    {
        if !hash_bodies && !self.tcx.sess.opts.build_dep_graph() {
            // If we just need the hashes in order to compute the SVH, we don't
            // need have two hashes per item. Just the one containing also the
            // item's body is sufficient.
            return
        }

        let def_path_hash = self.hcx.local_def_path_hash(def_index);

        let mut hasher = IchHasher::new();
        self.hcx.while_hashing_hir_bodies(hash_bodies, |hcx| {
            item_like.hash_stable(hcx, &mut hasher);
        });

        let bytes_hashed = hasher.bytes_hashed();
        let item_hash = hasher.finish();
        let dep_node = if hash_bodies {
            def_path_hash.to_dep_node(DepKind::HirBody)
        } else {
            def_path_hash.to_dep_node(DepKind::Hir)
        };
        debug!("calculate_def_hash: dep_node={:?} hash={:?}", dep_node, item_hash);
        self.hashes.insert(dep_node, item_hash);

        let bytes_hashed =
            self.tcx.sess.perf_stats.incr_comp_bytes_hashed.get() + bytes_hashed;
        self.tcx.sess.perf_stats.incr_comp_bytes_hashed.set(bytes_hashed);

        if hash_bodies {
            let in_scope_traits_map = self.tcx.in_scope_traits_map(def_index);
            let mut hasher = IchHasher::new();
            in_scope_traits_map.hash_stable(&mut self.hcx, &mut hasher);
            let dep_node = def_path_hash.to_dep_node(DepKind::InScopeTraits);
            self.hashes.insert(dep_node, hasher.finish());
        }
    }

    fn compute_crate_hash(&mut self) {
        let krate = self.tcx.hir.krate();

        let mut crate_state = IchHasher::new();

        let crate_disambiguator = self.tcx.sess.local_crate_disambiguator();
        "crate_disambiguator".hash(&mut crate_state);
        crate_disambiguator.as_str().len().hash(&mut crate_state);
        crate_disambiguator.as_str().hash(&mut crate_state);

        // add each item (in some deterministic order) to the overall
        // crate hash.
        {
            let mut item_hashes: Vec<_> =
                self.hashes.iter()
                           .filter_map(|(&item_dep_node, &item_hash)| {
                                // This `match` determines what kinds of nodes
                                // go into the SVH:
                                match item_dep_node.kind {
                                    DepKind::InScopeTraits |
                                    DepKind::Hir |
                                    DepKind::HirBody => {
                                        // We want to incoporate these into the
                                        // SVH.
                                    }
                                    DepKind::AllLocalTraitImpls => {
                                        // These are already covered by hashing
                                        // the HIR.
                                        return None
                                    }
                                    ref other => {
                                        bug!("Found unexpected DepKind during \
                                              SVH computation: {:?}",
                                             other)
                                    }
                                }

                                Some((item_dep_node, item_hash))
                           })
                           .collect();
            item_hashes.sort_unstable(); // avoid artificial dependencies on item ordering
            item_hashes.hash(&mut crate_state);
        }

        krate.attrs.hash_stable(&mut self.hcx, &mut crate_state);

        let crate_hash = crate_state.finish();
        self.hashes.insert(DepNode::new_no_params(DepKind::Krate), crate_hash);
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

        self.compute_and_store_ich_for_item_like(CRATE_DEF_INDEX,
                                                 false,
                                                 (module, (span, attrs)));
        self.compute_and_store_ich_for_item_like(CRATE_DEF_INDEX,
                                                 true,
                                                 (module, (span, attrs)));
    }

    fn compute_and_store_ich_for_trait_impls(&mut self, krate: &'tcx hir::Crate)
    {
        let tcx = self.tcx;

        let mut impls: Vec<(DefPathHash, Fingerprint)> = krate
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

        self.hashes.insert(DepNode::new_no_params(DepKind::AllLocalTraitImpls),
                           hasher.finish());
    }
}

impl<'a, 'tcx: 'a> ItemLikeVisitor<'tcx> for ComputeItemHashesVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let def_index = self.tcx.hir.local_def_id(item.id).index;
        self.compute_and_store_ich_for_item_like(def_index,
                                                 false,
                                                 item);
        self.compute_and_store_ich_for_item_like(def_index,
                                                 true,
                                                 item);
    }

    fn visit_trait_item(&mut self, item: &'tcx hir::TraitItem) {
        let def_index = self.tcx.hir.local_def_id(item.id).index;
        self.compute_and_store_ich_for_item_like(def_index,
                                                 false,
                                                 item);
        self.compute_and_store_ich_for_item_like(def_index,
                                                 true,
                                                 item);
    }

    fn visit_impl_item(&mut self, item: &'tcx hir::ImplItem) {
        let def_index = self.tcx.hir.local_def_id(item.id).index;
        self.compute_and_store_ich_for_item_like(def_index,
                                                 false,
                                                 item);
        self.compute_and_store_ich_for_item_like(def_index,
                                                 true,
                                                 item);
    }
}



pub fn compute_incremental_hashes_map<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                                    -> IncrementalHashesMap {
    let _ignore = tcx.dep_graph.in_ignore();
    let krate = tcx.hir.krate();

    let mut visitor = ComputeItemHashesVisitor {
        tcx,
        hcx: tcx.create_stable_hashing_context(),
        hashes: IncrementalHashesMap::new(),
    };

    record_time(&tcx.sess.perf_stats.incr_comp_hashes_time, || {
        visitor.hash_crate_root_module(krate);
        krate.visit_all_item_likes(&mut visitor);

        for macro_def in krate.exported_macros.iter() {
            let def_index = tcx.hir.local_def_id(macro_def.id).index;
            visitor.compute_and_store_ich_for_item_like(def_index,
                                                        false,
                                                        macro_def);
            visitor.compute_and_store_ich_for_item_like(def_index,
                                                        true,
                                                        macro_def);
        }

        visitor.compute_and_store_ich_for_trait_impls(krate);
    });

    tcx.sess.perf_stats.incr_comp_hashes_count.set(visitor.hashes.len() as u64);

    record_time(&tcx.sess.perf_stats.svh_time, || visitor.compute_crate_hash());
    visitor.hashes
}
