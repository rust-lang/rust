// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code to convert a DefId into a DefPath (when serializing) and then
//! back again (when deserializing). Note that the new DefId
//! necessarily will not be the same as the old (and of course the
//! item might even be removed in the meantime).

use rustc::dep_graph::DepNode;
use rustc::hir::map::DefPath;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::ty::TyCtxt;
use rustc::util::nodemap::DefIdMap;
use std::fmt::{self, Debug};
use std::iter::once;
use std::collections::HashMap;

/// Index into the DefIdDirectory
#[derive(Copy, Clone, Debug, PartialOrd, Ord, Hash, PartialEq, Eq,
         RustcEncodable, RustcDecodable)]
pub struct DefPathIndex {
    index: u32
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct DefIdDirectory {
    // N.B. don't use Removable here because these def-ids are loaded
    // directly without remapping, so loading them should not fail.
    paths: Vec<DefPath>,

    // For each crate, saves the crate-name/disambiguator so that
    // later we can match crate-numbers up again.
    krates: Vec<CrateInfo>,
}

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct CrateInfo {
    krate: CrateNum,
    name: String,
    disambiguator: String,
}

impl DefIdDirectory {
    pub fn new(krates: Vec<CrateInfo>) -> DefIdDirectory {
        DefIdDirectory { paths: vec![], krates: krates }
    }

    fn max_current_crate(&self, tcx: TyCtxt) -> CrateNum {
        tcx.sess.cstore.crates()
                       .into_iter()
                       .max()
                       .unwrap_or(LOCAL_CRATE)
    }

    /// Returns a string form for `index`; useful for debugging
    pub fn def_path_string(&self, tcx: TyCtxt, index: DefPathIndex) -> String {
        let path = &self.paths[index.index as usize];
        if self.krate_still_valid(tcx, self.max_current_crate(tcx), path.krate) {
            path.to_string(tcx)
        } else {
            format!("<crate {} changed>", path.krate)
        }
    }

    pub fn krate_still_valid(&self,
                             tcx: TyCtxt,
                             max_current_crate: CrateNum,
                             krate: CrateNum) -> bool {
        // Check that the crate-number still matches. For now, if it
        // doesn't, just return None. We could do better, such as
        // finding the new number.

        if krate > max_current_crate {
            false
        } else {
            let old_info = &self.krates[krate.as_usize()];
            assert_eq!(old_info.krate, krate);
            let old_name: &str = &old_info.name;
            let old_disambiguator: &str = &old_info.disambiguator;
            let new_name: &str = &tcx.crate_name(krate).as_str();
            let new_disambiguator: &str = &tcx.crate_disambiguator(krate).as_str();
            old_name == new_name && old_disambiguator == new_disambiguator
        }
    }

    pub fn retrace(&self, tcx: TyCtxt) -> RetracedDefIdDirectory {

        fn make_key(name: &str, disambiguator: &str) -> String {
            format!("{}/{}", name, disambiguator)
        }

        let new_krates: HashMap<_, _> =
            once(LOCAL_CRATE)
            .chain(tcx.sess.cstore.crates())
            .map(|krate| (make_key(&tcx.crate_name(krate).as_str(),
                                   &tcx.crate_disambiguator(krate).as_str()), krate))
            .collect();

        let ids = self.paths.iter()
                            .map(|path| {
                                let old_krate_id = path.krate.as_usize();
                                assert!(old_krate_id < self.krates.len());
                                let old_crate_info = &self.krates[old_krate_id];
                                let old_crate_key = make_key(&old_crate_info.name,
                                                         &old_crate_info.disambiguator);
                                if let Some(&new_crate_key) = new_krates.get(&old_crate_key) {
                                    tcx.retrace_path(new_crate_key, &path.data)
                                } else {
                                    debug!("crate {:?} no longer exists", old_crate_key);
                                    None
                                }
                            })
                            .collect();
        RetracedDefIdDirectory { ids: ids }
    }
}

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct RetracedDefIdDirectory {
    ids: Vec<Option<DefId>>
}

impl RetracedDefIdDirectory {
    pub fn def_id(&self, index: DefPathIndex) -> Option<DefId> {
        self.ids[index.index as usize]
    }

    pub fn map(&self, node: &DepNode<DefPathIndex>) -> Option<DepNode<DefId>> {
        node.map_def(|&index| self.def_id(index))
    }
}

pub struct DefIdDirectoryBuilder<'a,'tcx:'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    hash: DefIdMap<DefPathIndex>,
    directory: DefIdDirectory,
}

impl<'a,'tcx> DefIdDirectoryBuilder<'a,'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> DefIdDirectoryBuilder<'a, 'tcx> {
        let mut krates: Vec<_> =
            once(LOCAL_CRATE)
            .chain(tcx.sess.cstore.crates())
            .map(|krate| {
                CrateInfo {
                    krate: krate,
                    name: tcx.crate_name(krate).to_string(),
                    disambiguator: tcx.crate_disambiguator(krate).to_string()
                }
            })
            .collect();

        // the result of crates() is not in order, so sort list of
        // crates so that we can just index it later
        krates.sort_by_key(|k| k.krate);

        DefIdDirectoryBuilder {
            tcx: tcx,
            hash: DefIdMap(),
            directory: DefIdDirectory::new(krates),
        }
    }

    pub fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.tcx
    }

    pub fn add(&mut self, def_id: DefId) -> DefPathIndex {
        debug!("DefIdDirectoryBuilder: def_id={:?}", def_id);
        let tcx = self.tcx;
        let paths = &mut self.directory.paths;
        self.hash.entry(def_id)
                 .or_insert_with(|| {
                     let def_path = tcx.def_path(def_id);
                     let index = paths.len() as u32;
                     paths.push(def_path);
                     DefPathIndex { index: index }
                 })
                 .clone()
    }

    pub fn lookup_def_path(&self, id: DefPathIndex) -> &DefPath {
        &self.directory.paths[id.index as usize]
    }

    pub fn map(&mut self, node: &DepNode<DefId>) -> DepNode<DefPathIndex> {
        node.map_def(|&def_id| Some(self.add(def_id))).unwrap()
    }

    pub fn directory(&self) -> &DefIdDirectory {
        &self.directory
    }
}

impl Debug for DefIdDirectory {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.debug_list()
           .entries(self.paths.iter().enumerate())
           .finish()
    }
}
