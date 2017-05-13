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

use syntax::ast;
use std::cell::RefCell;
use std::hash::Hash;
use rustc::dep_graph::DepNode;
use rustc::hir;
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::hir::intravisit as visit;
use rustc::hir::intravisit::{Visitor, NestedVisitorMap};
use rustc::ty::TyCtxt;
use rustc_data_structures::stable_hasher::StableHasher;
use ich::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc::util::common::record_time;
use rustc::session::config::DebugInfoLevel::NoDebugInfo;

use self::def_path_hash::DefPathHashes;
use self::svh_visitor::StrictVersionHashVisitor;
use self::caching_codemap_view::CachingCodemapView;

mod def_path_hash;
mod svh_visitor;
mod caching_codemap_view;

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


pub fn compute_incremental_hashes_map<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                                    -> IncrementalHashesMap {
    let _ignore = tcx.dep_graph.in_ignore();
    let krate = tcx.map.krate();
    let hash_spans = tcx.sess.opts.debuginfo != NoDebugInfo;
    let mut visitor = HashItemsVisitor {
        tcx: tcx,
        hashes: IncrementalHashesMap::new(),
        def_path_hashes: DefPathHashes::new(tcx),
        codemap: CachingCodemapView::new(tcx),
        hash_spans: hash_spans,
    };
    record_time(&tcx.sess.perf_stats.incr_comp_hashes_time, || {
        visitor.calculate_def_id(DefId::local(CRATE_DEF_INDEX), |v| {
            v.hash_crate_root_module(krate);
        });
        krate.visit_all_item_likes(&mut visitor.as_deep_visitor());

        for macro_def in krate.exported_macros.iter() {
            visitor.calculate_node_id(macro_def.id,
                                      |v| v.visit_macro_def(macro_def));
        }
    });

    tcx.sess.perf_stats.incr_comp_hashes_count.set(visitor.hashes.len() as u64);

    record_time(&tcx.sess.perf_stats.svh_time, || visitor.compute_crate_hash());
    visitor.hashes
}

struct HashItemsVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    def_path_hashes: DefPathHashes<'a, 'tcx>,
    codemap: CachingCodemapView<'tcx>,
    hashes: IncrementalHashesMap,
    hash_spans: bool,
}

impl<'a, 'tcx> HashItemsVisitor<'a, 'tcx> {
    fn calculate_node_id<W>(&mut self, id: ast::NodeId, walk_op: W)
        where W: for<'v> FnMut(&mut StrictVersionHashVisitor<'v, 'a, 'tcx>)
    {
        let def_id = self.tcx.map.local_def_id(id);
        self.calculate_def_id(def_id, walk_op)
    }

    fn calculate_def_id<W>(&mut self, def_id: DefId, mut walk_op: W)
        where W: for<'v> FnMut(&mut StrictVersionHashVisitor<'v, 'a, 'tcx>)
    {
        assert!(def_id.is_local());
        debug!("HashItemsVisitor::calculate(def_id={:?})", def_id);
        self.calculate_def_hash(DepNode::Hir(def_id), false, &mut walk_op);
        self.calculate_def_hash(DepNode::HirBody(def_id), true, &mut walk_op);
    }

    fn calculate_def_hash<W>(&mut self,
                             dep_node: DepNode<DefId>,
                             hash_bodies: bool,
                             walk_op: &mut W)
        where W: for<'v> FnMut(&mut StrictVersionHashVisitor<'v, 'a, 'tcx>)
    {
        let mut state = IchHasher::new();
        walk_op(&mut StrictVersionHashVisitor::new(&mut state,
                                                   self.tcx,
                                                   &mut self.def_path_hashes,
                                                   &mut self.codemap,
                                                   self.hash_spans,
                                                   hash_bodies));
        let bytes_hashed = state.bytes_hashed();
        let item_hash = state.finish();
        debug!("calculate_def_hash: dep_node={:?} hash={:?}", dep_node, item_hash);
        self.hashes.insert(dep_node, item_hash);

        let bytes_hashed = self.tcx.sess.perf_stats.incr_comp_bytes_hashed.get() +
            bytes_hashed;
        self.tcx.sess.perf_stats.incr_comp_bytes_hashed.set(bytes_hashed);
    }

    fn compute_crate_hash(&mut self) {
        let krate = self.tcx.map.krate();

        let mut crate_state = IchHasher::new();

        let crate_disambiguator = self.tcx.sess.local_crate_disambiguator();
        "crate_disambiguator".hash(&mut crate_state);
        crate_disambiguator.as_str().len().hash(&mut crate_state);
        crate_disambiguator.as_str().hash(&mut crate_state);

        // add each item (in some deterministic order) to the overall
        // crate hash.
        {
            let def_path_hashes = &mut self.def_path_hashes;
            let mut item_hashes: Vec<_> =
                self.hashes.iter()
                           .map(|(item_dep_node, &item_hash)| {
                               // convert from a DepNode<DefId> tp a
                               // DepNode<u64> where the u64 is the
                               // hash of the def-id's def-path:
                               let item_dep_node =
                                   item_dep_node.map_def(|&did| Some(def_path_hashes.hash(did)))
                                                .unwrap();
                               (item_dep_node, item_hash)
                           })
                           .collect();
            item_hashes.sort(); // avoid artificial dependencies on item ordering
            item_hashes.hash(&mut crate_state);
        }

        {
            let mut visitor = StrictVersionHashVisitor::new(&mut crate_state,
                                                            self.tcx,
                                                            &mut self.def_path_hashes,
                                                            &mut self.codemap,
                                                            self.hash_spans,
                                                            false);
            visitor.hash_attributes(&krate.attrs);
        }

        let crate_hash = crate_state.finish();
        self.hashes.insert(DepNode::Krate, crate_hash);
        debug!("calculate_crate_hash: crate_hash={:?}", crate_hash);
    }
}


impl<'a, 'tcx> Visitor<'tcx> for HashItemsVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        self.calculate_node_id(item.id, |v| v.visit_item(item));
        visit::walk_item(self, item);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {
        self.calculate_node_id(trait_item.id, |v| v.visit_trait_item(trait_item));
        visit::walk_trait_item(self, trait_item);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        self.calculate_node_id(impl_item.id, |v| v.visit_impl_item(impl_item));
        visit::walk_impl_item(self, impl_item);
    }
}
