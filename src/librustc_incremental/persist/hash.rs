// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::dep_graph::{DepNode, DepKind};
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::hir::svh::Svh;
use rustc::ich::Fingerprint;
use rustc::ty::TyCtxt;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::flock;
use rustc_serialize::Decodable;
use rustc_serialize::opaque::Decoder;

use IncrementalHashesMap;
use super::data::*;
use super::fs::*;
use super::file_format;

use std::hash::Hash;
use std::fmt::Debug;

pub struct HashContext<'a, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    incremental_hashes_map: &'a IncrementalHashesMap,
    metadata_hashes: FxHashMap<DefId, Fingerprint>,
    crate_hashes: FxHashMap<CrateNum, Svh>,
}

impl<'a, 'tcx> HashContext<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
               incremental_hashes_map: &'a IncrementalHashesMap)
               -> Self {
        HashContext {
            tcx,
            incremental_hashes_map,
            metadata_hashes: FxHashMap(),
            crate_hashes: FxHashMap(),
        }
    }

    pub fn is_hashable(tcx: TyCtxt, dep_node: &DepNode) -> bool {
        match dep_node.kind {
            DepKind::Krate |
            DepKind::Hir |
            DepKind::InScopeTraits |
            DepKind::HirBody =>
                true,
            DepKind::MetaData => {
                let def_id = dep_node.extract_def_id(tcx).unwrap();
                !def_id.is_local()
            }
            _ => false,
        }
    }

    pub fn hash(&mut self, dep_node: &DepNode) -> Option<Fingerprint> {
        match dep_node.kind {
            DepKind::Krate => {
                Some(self.incremental_hashes_map[dep_node])
            }

            // HIR nodes (which always come from our crate) are an input:
            DepKind::InScopeTraits |
            DepKind::Hir |
            DepKind::HirBody => {
                Some(self.incremental_hashes_map[dep_node])
            }

            // MetaData from other crates is an *input* to us.
            // MetaData nodes from *our* crates are an *output*; we
            // don't hash them, but we do compute a hash for them and
            // save it for others to use.
            DepKind::MetaData => {
                let def_id = dep_node.extract_def_id(self.tcx).unwrap();
                if !def_id.is_local() {
                    Some(self.metadata_hash(def_id,
                                        def_id.krate,
                                        |this| &mut this.metadata_hashes))
                } else {
                    None
                }
            }

            _ => {
                // Other kinds of nodes represent computed by-products
                // that we don't hash directly; instead, they should
                // have some transitive dependency on a Hir or
                // MetaData node, so we'll just hash that
                None
            }
        }
    }

    fn metadata_hash<K, C>(&mut self,
                           key: K,
                           cnum: CrateNum,
                           cache: C)
                           -> Fingerprint
        where K: Hash + Eq + Debug,
              C: Fn(&mut Self) -> &mut FxHashMap<K, Fingerprint>,
    {
        debug!("metadata_hash(key={:?})", key);

        debug_assert!(cnum != LOCAL_CRATE);
        loop {
            // check whether we have a result cached for this def-id
            if let Some(&hash) = cache(self).get(&key) {
                return hash;
            }

            // check whether we did not find detailed metadata for this
            // krate; in that case, we just use the krate's overall hash
            if let Some(&svh) = self.crate_hashes.get(&cnum) {
                // micro-"optimization": avoid a cache miss if we ask
                // for metadata from this particular def-id again.
                let fingerprint = svh_to_fingerprint(svh);
                cache(self).insert(key, fingerprint);

                return fingerprint;
            }

            // otherwise, load the data and repeat.
            self.load_data(cnum);
            assert!(self.crate_hashes.contains_key(&cnum));
        }
    }

    fn load_data(&mut self, cnum: CrateNum) {
        debug!("load_data(cnum={})", cnum);

        let svh = self.tcx.crate_hash(cnum);
        let old = self.crate_hashes.insert(cnum, svh);
        debug!("load_data: svh={}", svh);
        assert!(old.is_none(), "loaded data for crate {:?} twice", cnum);

        if let Some(session_dir) = find_metadata_hashes_for(self.tcx, cnum) {
            debug!("load_data: session_dir={:?}", session_dir);

            // Lock the directory we'll be reading  the hashes from.
            let lock_file_path = lock_file_path(&session_dir);
            let _lock = match flock::Lock::new(&lock_file_path,
                                               false,   // don't wait
                                               false,   // don't create the lock-file
                                               false) { // shared lock
                Ok(lock) => lock,
                Err(err) => {
                    debug!("Could not acquire lock on `{}` while trying to \
                            load metadata hashes: {}",
                            lock_file_path.display(),
                            err);

                    // Could not acquire the lock. The directory is probably in
                    // in the process of being deleted. It's OK to just exit
                    // here. It's the same scenario as if the file had not
                    // existed in the first place.
                    return
                }
            };

            let hashes_file_path = metadata_hash_import_path(&session_dir);

            match file_format::read_file(self.tcx.sess, &hashes_file_path)
            {
                Ok(Some(data)) => {
                    match self.load_from_data(cnum, &data, svh) {
                        Ok(()) => { }
                        Err(err) => {
                            bug!("decoding error in dep-graph from `{}`: {}",
                                 &hashes_file_path.display(), err);
                        }
                    }
                }
                Ok(None) => {
                    // If the file is not found, that's ok.
                }
                Err(err) => {
                    self.tcx.sess.err(
                        &format!("could not load dep information from `{}`: {}",
                                 hashes_file_path.display(), err));
                }
            }
        }
    }

    fn load_from_data(&mut self,
                      cnum: CrateNum,
                      data: &[u8],
                      expected_svh: Svh) -> Result<(), String> {
        debug!("load_from_data(cnum={})", cnum);

        // Load up the hashes for the def-ids from this crate.
        let mut decoder = Decoder::new(data, 0);
        let svh_in_hashes_file = Svh::decode(&mut decoder)?;

        if svh_in_hashes_file != expected_svh {
            // We should not be able to get here. If we do, then
            // `fs::find_metadata_hashes_for()` has messed up.
            bug!("mismatch between SVH in crate and SVH in incr. comp. hashes")
        }

        let serialized_hashes = SerializedMetadataHashes::decode(&mut decoder)?;
        for serialized_hash in serialized_hashes.entry_hashes {
            // the hashes are stored with just a def-index, which is
            // always relative to the old crate; convert that to use
            // our internal crate number
            let def_id = DefId { krate: cnum, index: serialized_hash.def_index };

            // record the hash for this dep-node
            let old = self.metadata_hashes.insert(def_id, serialized_hash.hash);
            debug!("load_from_data: def_id={:?} hash={}", def_id, serialized_hash.hash);
            assert!(old.is_none(), "already have hash for {:?}", def_id);
        }

        Ok(())
    }
}

fn svh_to_fingerprint(svh: Svh) -> Fingerprint {
    Fingerprint::from_smaller_hash(svh.as_u64())
}
