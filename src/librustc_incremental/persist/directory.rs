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
use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc::util::nodemap::DefIdMap;
use rustc_serialize::{Decoder as RustcDecoder, Encoder as RustcEncoder};
use std::fmt::{self, Debug};

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
    paths: Vec<DefPath>
}

impl DefIdDirectory {
    pub fn new() -> DefIdDirectory {
        DefIdDirectory { paths: vec![] }
    }

    pub fn retrace(&self, tcx: &ty::TyCtxt) -> RetracedDefIdDirectory {
        let ids = self.paths.iter()
                            .map(|path| tcx.map.retrace_path(path))
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

    pub fn map(&self, node: DepNode<DefPathIndex>) -> Option<DepNode<DefId>> {
        node.map_def(|&index| self.def_id(index))
    }
}

pub struct DefIdDirectoryBuilder<'a,'tcx:'a> {
    tcx: &'a ty::TyCtxt<'tcx>,
    hash: DefIdMap<Option<DefPathIndex>>,
    directory: DefIdDirectory,
}

impl<'a,'tcx> DefIdDirectoryBuilder<'a,'tcx> {
    pub fn new(tcx: &'a ty::TyCtxt<'tcx>) -> DefIdDirectoryBuilder<'a, 'tcx> {
        DefIdDirectoryBuilder {
            tcx: tcx,
            hash: DefIdMap(),
            directory: DefIdDirectory::new()
        }
    }

    pub fn add(&mut self, def_id: DefId) -> Option<DefPathIndex> {
        if !def_id.is_local() {
            // FIXME(#32015) clarify story about cross-crate dep tracking
            return None;
        }

        let tcx = self.tcx;
        let paths = &mut self.directory.paths;
        self.hash.entry(def_id)
                 .or_insert_with(|| {
                     let def_path = tcx.def_path(def_id);
                     if !def_path.is_local() {
                         return None;
                     }
                     let index = paths.len() as u32;
                     paths.push(def_path);
                     Some(DefPathIndex { index: index })
                 })
                 .clone()
    }

    pub fn map(&mut self, node: DepNode<DefId>) -> Option<DepNode<DefPathIndex>> {
        node.map_def(|&def_id| self.add(def_id))
    }

    pub fn into_directory(self) -> DefIdDirectory {
        self.directory
    }
}

impl Debug for DefIdDirectory {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.debug_list()
           .entries(self.paths.iter().enumerate())
           .finish()
    }
}
