// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The crate store - a central repo for information collected about external
// crates and libraries

use schema;
use rustc::hir::def_id::{CrateNum, DefIndex};
use rustc::hir::map::definitions::DefPathTable;
use rustc::middle::cstore::{DepKind, ExternCrate, MetadataLoader};
use rustc::mir::interpret::AllocDecodingState;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc::util::nodemap::{FxHashMap, NodeMap};

use rustc_data_structures::sync::{Lrc, RwLock, Lock};
use syntax::ast;
use syntax::ext::base::SyntaxExtension;
use syntax::symbol::Symbol;
use syntax_pos;

pub use rustc::middle::cstore::{NativeLibrary, NativeLibraryKind, LinkagePreference};
pub use rustc::middle::cstore::NativeLibraryKind::*;
pub use rustc::middle::cstore::{CrateSource, LibSource, ForeignModule};

pub use cstore_impl::{provide, provide_extern};

// A map from external crate numbers (as decoded from some crate file) to
// local crate numbers (as generated during this session). Each external
// crate may refer to types in other external crates, and each has their
// own crate numbers.
pub type CrateNumMap = IndexVec<CrateNum, CrateNum>;

pub use rustc_data_structures::sync::MetadataRef;

pub struct MetadataBlob(pub MetadataRef);

/// Holds information about a syntax_pos::FileMap imported from another crate.
/// See `imported_filemaps()` for more information.
pub struct ImportedFileMap {
    /// This FileMap's byte-offset within the codemap of its original crate
    pub original_start_pos: syntax_pos::BytePos,
    /// The end of this FileMap within the codemap of its original crate
    pub original_end_pos: syntax_pos::BytePos,
    /// The imported FileMap's representation within the local codemap
    pub translated_filemap: Lrc<syntax_pos::FileMap>,
}

pub struct CrateMetadata {
    pub name: Symbol,

    /// Information about the extern crate that caused this crate to
    /// be loaded. If this is `None`, then the crate was injected
    /// (e.g., by the allocator)
    pub extern_crate: Lock<Option<ExternCrate>>,

    pub blob: MetadataBlob,
    pub cnum_map: CrateNumMap,
    pub cnum: CrateNum,
    pub dependencies: Lock<Vec<CrateNum>>,
    pub codemap_import_info: RwLock<Vec<ImportedFileMap>>,

    /// Used for decoding interpret::AllocIds in a cached & thread-safe manner.
    pub alloc_decoding_state: AllocDecodingState,

    pub root: schema::CrateRoot,

    /// For each public item in this crate, we encode a key.  When the
    /// crate is loaded, we read all the keys and put them in this
    /// hashmap, which gives the reverse mapping.  This allows us to
    /// quickly retrace a `DefPath`, which is needed for incremental
    /// compilation support.
    pub def_path_table: Lrc<DefPathTable>,

    pub trait_impls: FxHashMap<(u32, DefIndex), schema::LazySeq<DefIndex>>,

    pub dep_kind: Lock<DepKind>,
    pub source: CrateSource,

    pub proc_macros: Option<Vec<(ast::Name, Lrc<SyntaxExtension>)>>,
}

pub struct CStore {
    metas: RwLock<IndexVec<CrateNum, Option<Lrc<CrateMetadata>>>>,
    /// Map from NodeId's of local extern crate statements to crate numbers
    extern_mod_crate_map: Lock<NodeMap<CrateNum>>,
    pub metadata_loader: Box<MetadataLoader + Sync>,
}

impl CStore {
    pub fn new(metadata_loader: Box<MetadataLoader + Sync>) -> CStore {
        CStore {
            // We add an empty entry for LOCAL_CRATE (which maps to zero) in
            // order to make array indices in `metas` match with the
            // corresponding `CrateNum`. This first entry will always remain
            // `None`.
            metas: RwLock::new(IndexVec::from_elem_n(None, 1)),
            extern_mod_crate_map: Lock::new(FxHashMap()),
            metadata_loader,
        }
    }

    pub(super) fn alloc_new_crate_num(&self) -> CrateNum {
        let mut metas = self.metas.borrow_mut();
        let cnum = CrateNum::new(metas.len());
        metas.push(None);
        cnum
    }

    pub(super) fn get_crate_data(&self, cnum: CrateNum) -> Lrc<CrateMetadata> {
        self.metas.borrow()[cnum].clone().unwrap()
    }

    pub(super) fn set_crate_data(&self, cnum: CrateNum, data: Lrc<CrateMetadata>) {
        let mut metas = self.metas.borrow_mut();
        assert!(metas[cnum].is_none(), "Overwriting crate metadata entry");
        metas[cnum] = Some(data);
    }

    pub(super) fn iter_crate_data<I>(&self, mut i: I)
        where I: FnMut(CrateNum, &Lrc<CrateMetadata>)
    {
        for (k, v) in self.metas.borrow().iter_enumerated() {
            if let &Some(ref v) = v {
                i(k, v);
            }
        }
    }

    pub(super) fn crate_dependencies_in_rpo(&self, krate: CrateNum) -> Vec<CrateNum> {
        let mut ordering = Vec::new();
        self.push_dependencies_in_postorder(&mut ordering, krate);
        ordering.reverse();
        ordering
    }

    pub(super) fn push_dependencies_in_postorder(&self,
                                                 ordering: &mut Vec<CrateNum>,
                                                 krate: CrateNum) {
        if ordering.contains(&krate) {
            return;
        }

        let data = self.get_crate_data(krate);
        for &dep in data.dependencies.borrow().iter() {
            if dep != krate {
                self.push_dependencies_in_postorder(ordering, dep);
            }
        }

        ordering.push(krate);
    }

    pub(super) fn do_postorder_cnums_untracked(&self) -> Vec<CrateNum> {
        let mut ordering = Vec::new();
        for (num, v) in self.metas.borrow().iter_enumerated() {
            if let &Some(_) = v {
                self.push_dependencies_in_postorder(&mut ordering, num);
            }
        }
        return ordering
    }

    pub(super) fn add_extern_mod_stmt_cnum(&self, emod_id: ast::NodeId, cnum: CrateNum) {
        self.extern_mod_crate_map.borrow_mut().insert(emod_id, cnum);
    }

    pub(super) fn do_extern_mod_stmt_cnum(&self, emod_id: ast::NodeId) -> Option<CrateNum> {
        self.extern_mod_crate_map.borrow().get(&emod_id).cloned()
    }
}
