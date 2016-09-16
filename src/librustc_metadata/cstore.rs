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

use loader;
use schema;

use rustc::dep_graph::DepGraph;
use rustc::hir::def_id::{CRATE_DEF_INDEX, CrateNum, DefIndex, DefId};
use rustc::hir::map::DefKey;
use rustc::hir::svh::Svh;
use rustc::middle::cstore::ExternCrate;
use rustc::session::config::PanicStrategy;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc::util::nodemap::{FnvHashMap, NodeMap, NodeSet, DefIdMap, FnvHashSet};

use std::cell::{RefCell, Cell};
use std::rc::Rc;
use std::path::PathBuf;
use flate::Bytes;
use syntax::ast::{self, Ident};
use syntax::attr;
use syntax_pos;

pub use rustc::middle::cstore::{NativeLibraryKind, LinkagePreference};
pub use rustc::middle::cstore::{NativeStatic, NativeFramework, NativeUnknown};
pub use rustc::middle::cstore::{CrateSource, LinkMeta};

// A map from external crate numbers (as decoded from some crate file) to
// local crate numbers (as generated during this session). Each external
// crate may refer to types in other external crates, and each has their
// own crate numbers.
pub type CrateNumMap = IndexVec<CrateNum, CrateNum>;

pub enum MetadataBlob {
    Inflated(Bytes),
    Archive(loader::ArchiveMetadata),
}

/// Holds information about a syntax_pos::FileMap imported from another crate.
/// See `imported_filemaps()` for more information.
pub struct ImportedFileMap {
    /// This FileMap's byte-offset within the codemap of its original crate
    pub original_start_pos: syntax_pos::BytePos,
    /// The end of this FileMap within the codemap of its original crate
    pub original_end_pos: syntax_pos::BytePos,
    /// The imported FileMap's representation within the local codemap
    pub translated_filemap: Rc<syntax_pos::FileMap>
}

pub struct CrateMetadata {
    pub name: String,

    /// Information about the extern crate that caused this crate to
    /// be loaded. If this is `None`, then the crate was injected
    /// (e.g., by the allocator)
    pub extern_crate: Cell<Option<ExternCrate>>,

    pub blob: MetadataBlob,
    pub cnum_map: RefCell<CrateNumMap>,
    pub cnum: CrateNum,
    pub codemap_import_info: RefCell<Vec<ImportedFileMap>>,

    pub root: schema::CrateRoot,

    /// For each public item in this crate, we encode a key.  When the
    /// crate is loaded, we read all the keys and put them in this
    /// hashmap, which gives the reverse mapping.  This allows us to
    /// quickly retrace a `DefPath`, which is needed for incremental
    /// compilation support.
    pub key_map: FnvHashMap<DefKey, DefIndex>,

    /// Flag if this crate is required by an rlib version of this crate, or in
    /// other words whether it was explicitly linked to. An example of a crate
    /// where this is false is when an allocator crate is injected into the
    /// dependency list, and therefore isn't actually needed to link an rlib.
    pub explicitly_linked: Cell<bool>,
}

pub struct CachedInlinedItem {
    /// The NodeId of the RootInlinedParent HIR map entry
    pub inlined_root: ast::NodeId,
    /// The local NodeId of the inlined entity
    pub item_id: ast::NodeId,
}

pub struct CStore {
    pub dep_graph: DepGraph,
    metas: RefCell<FnvHashMap<CrateNum, Rc<CrateMetadata>>>,
    /// Map from NodeId's of local extern crate statements to crate numbers
    extern_mod_crate_map: RefCell<NodeMap<CrateNum>>,
    used_crate_sources: RefCell<Vec<CrateSource>>,
    used_libraries: RefCell<Vec<(String, NativeLibraryKind)>>,
    used_link_args: RefCell<Vec<String>>,
    statically_included_foreign_items: RefCell<NodeSet>,
    pub inlined_item_cache: RefCell<DefIdMap<Option<CachedInlinedItem>>>,
    pub defid_for_inlined_node: RefCell<NodeMap<DefId>>,
    pub visible_parent_map: RefCell<DefIdMap<DefId>>,
    pub used_for_derive_macro: RefCell<FnvHashSet<Ident>>,
}

impl CStore {
    pub fn new(dep_graph: &DepGraph) -> CStore {
        CStore {
            dep_graph: dep_graph.clone(),
            metas: RefCell::new(FnvHashMap()),
            extern_mod_crate_map: RefCell::new(FnvHashMap()),
            used_crate_sources: RefCell::new(Vec::new()),
            used_libraries: RefCell::new(Vec::new()),
            used_link_args: RefCell::new(Vec::new()),
            statically_included_foreign_items: RefCell::new(NodeSet()),
            visible_parent_map: RefCell::new(FnvHashMap()),
            inlined_item_cache: RefCell::new(FnvHashMap()),
            defid_for_inlined_node: RefCell::new(FnvHashMap()),
            used_for_derive_macro: RefCell::new(FnvHashSet()),
        }
    }

    pub fn next_crate_num(&self) -> CrateNum {
        CrateNum::new(self.metas.borrow().len() + 1)
    }

    pub fn get_crate_data(&self, cnum: CrateNum) -> Rc<CrateMetadata> {
        self.metas.borrow().get(&cnum).unwrap().clone()
    }

    pub fn get_crate_hash(&self, cnum: CrateNum) -> Svh {
        self.get_crate_data(cnum).hash()
    }

    pub fn set_crate_data(&self, cnum: CrateNum, data: Rc<CrateMetadata>) {
        self.metas.borrow_mut().insert(cnum, data);
    }

    pub fn iter_crate_data<I>(&self, mut i: I) where
        I: FnMut(CrateNum, &Rc<CrateMetadata>),
    {
        for (&k, v) in self.metas.borrow().iter() {
            i(k, v);
        }
    }

    /// Like `iter_crate_data`, but passes source paths (if available) as well.
    pub fn iter_crate_data_origins<I>(&self, mut i: I) where
        I: FnMut(CrateNum, &CrateMetadata, Option<CrateSource>),
    {
        for (&k, v) in self.metas.borrow().iter() {
            let origin = self.opt_used_crate_source(k);
            origin.as_ref().map(|cs| { assert!(k == cs.cnum); });
            i(k, &v, origin);
        }
    }

    pub fn add_used_crate_source(&self, src: CrateSource) {
        let mut used_crate_sources = self.used_crate_sources.borrow_mut();
        if !used_crate_sources.contains(&src) {
            used_crate_sources.push(src);
        }
    }

    pub fn opt_used_crate_source(&self, cnum: CrateNum)
                                 -> Option<CrateSource> {
        self.used_crate_sources.borrow_mut()
            .iter().find(|source| source.cnum == cnum).cloned()
    }

    pub fn reset(&self) {
        self.metas.borrow_mut().clear();
        self.extern_mod_crate_map.borrow_mut().clear();
        self.used_crate_sources.borrow_mut().clear();
        self.used_libraries.borrow_mut().clear();
        self.used_link_args.borrow_mut().clear();
        self.statically_included_foreign_items.borrow_mut().clear();
    }

    pub fn crate_dependencies_in_rpo(&self, krate: CrateNum) -> Vec<CrateNum>
    {
        let mut ordering = Vec::new();
        self.push_dependencies_in_postorder(&mut ordering, krate);
        ordering.reverse();
        ordering
    }

    pub fn push_dependencies_in_postorder(&self,
                                          ordering: &mut Vec<CrateNum>,
                                          krate: CrateNum)
    {
        if ordering.contains(&krate) { return }

        let data = self.get_crate_data(krate);
        for &dep in data.cnum_map.borrow().iter() {
            if dep != krate {
                self.push_dependencies_in_postorder(ordering, dep);
            }
        }

        ordering.push(krate);
    }

    // This method is used when generating the command line to pass through to
    // system linker. The linker expects undefined symbols on the left of the
    // command line to be defined in libraries on the right, not the other way
    // around. For more info, see some comments in the add_used_library function
    // below.
    //
    // In order to get this left-to-right dependency ordering, we perform a
    // topological sort of all crates putting the leaves at the right-most
    // positions.
    pub fn do_get_used_crates(&self, prefer: LinkagePreference)
                              -> Vec<(CrateNum, Option<PathBuf>)> {
        let mut ordering = Vec::new();
        for (&num, _) in self.metas.borrow().iter() {
            self.push_dependencies_in_postorder(&mut ordering, num);
        }
        info!("topological ordering: {:?}", ordering);
        ordering.reverse();
        let mut libs = self.used_crate_sources.borrow()
            .iter()
            .map(|src| (src.cnum, match prefer {
                LinkagePreference::RequireDynamic => src.dylib.clone().map(|p| p.0),
                LinkagePreference::RequireStatic => src.rlib.clone().map(|p| p.0),
            }))
            .collect::<Vec<_>>();
        libs.sort_by(|&(a, _), &(b, _)| {
            let a = ordering.iter().position(|x| *x == a);
            let b = ordering.iter().position(|x| *x == b);
            a.cmp(&b)
        });
        libs
    }

    pub fn add_used_library(&self, lib: String, kind: NativeLibraryKind) {
        assert!(!lib.is_empty());
        self.used_libraries.borrow_mut().push((lib, kind));
    }

    pub fn get_used_libraries<'a>(&'a self)
                              -> &'a RefCell<Vec<(String,
                                                  NativeLibraryKind)>> {
        &self.used_libraries
    }

    pub fn add_used_link_args(&self, args: &str) {
        for s in args.split(' ').filter(|s| !s.is_empty()) {
            self.used_link_args.borrow_mut().push(s.to_string());
        }
    }

    pub fn get_used_link_args<'a>(&'a self) -> &'a RefCell<Vec<String> > {
        &self.used_link_args
    }

    pub fn add_extern_mod_stmt_cnum(&self,
                                    emod_id: ast::NodeId,
                                    cnum: CrateNum) {
        self.extern_mod_crate_map.borrow_mut().insert(emod_id, cnum);
    }

    pub fn add_statically_included_foreign_item(&self, id: ast::NodeId) {
        self.statically_included_foreign_items.borrow_mut().insert(id);
    }

    pub fn do_is_statically_included_foreign_item(&self, id: ast::NodeId) -> bool {
        self.statically_included_foreign_items.borrow().contains(&id)
    }

    pub fn do_extern_mod_stmt_cnum(&self, emod_id: ast::NodeId) -> Option<CrateNum>
    {
        self.extern_mod_crate_map.borrow().get(&emod_id).cloned()
    }

    pub fn was_used_for_derive_macros(&self, i: &ast::Item) -> bool {
        self.used_for_derive_macro.borrow().contains(&i.ident)
    }

    pub fn add_used_for_derive_macros(&self, i: &ast::Item) {
        self.used_for_derive_macro.borrow_mut().insert(i.ident);
    }
}

impl CrateMetadata {
    pub fn name(&self) -> &str { &self.root.name }
    pub fn hash(&self) -> Svh { self.root.hash }
    pub fn disambiguator(&self) -> &str { &self.root.disambiguator }

    pub fn is_staged_api(&self) -> bool {
        self.get_item_attrs(CRATE_DEF_INDEX).iter().any(|attr| {
            attr.name() == "stable" || attr.name() == "unstable"
        })
    }

    pub fn is_allocator(&self) -> bool {
        let attrs = self.get_item_attrs(CRATE_DEF_INDEX);
        attr::contains_name(&attrs, "allocator")
    }

    pub fn needs_allocator(&self) -> bool {
        let attrs = self.get_item_attrs(CRATE_DEF_INDEX);
        attr::contains_name(&attrs, "needs_allocator")
    }

    pub fn is_panic_runtime(&self) -> bool {
        let attrs = self.get_item_attrs(CRATE_DEF_INDEX);
        attr::contains_name(&attrs, "panic_runtime")
    }

    pub fn needs_panic_runtime(&self) -> bool {
        let attrs = self.get_item_attrs(CRATE_DEF_INDEX);
        attr::contains_name(&attrs, "needs_panic_runtime")
    }

    pub fn is_compiler_builtins(&self) -> bool {
        let attrs = self.get_item_attrs(CRATE_DEF_INDEX);
        attr::contains_name(&attrs, "compiler_builtins")
    }

    pub fn is_no_builtins(&self) -> bool {
        let attrs = self.get_item_attrs(CRATE_DEF_INDEX);
        attr::contains_name(&attrs, "no_builtins")
    }

    pub fn panic_strategy(&self) -> PanicStrategy {
        self.root.panic_strategy.clone()
    }
}
