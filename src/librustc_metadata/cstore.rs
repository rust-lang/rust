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

use locator;
use schema;

use rustc::dep_graph::DepGraph;
use rustc::hir::def_id::{CRATE_DEF_INDEX, LOCAL_CRATE, CrateNum, DefIndex, DefId};
use rustc::hir::map::definitions::DefPathTable;
use rustc::hir::svh::Svh;
use rustc::middle::cstore::{DepKind, ExternCrate};
use rustc_back::PanicStrategy;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc::util::nodemap::{FxHashMap, FxHashSet, NodeMap, DefIdMap};

use std::cell::{RefCell, Cell};
use std::rc::Rc;
use flate::Bytes;
use syntax::{ast, attr};
use syntax::ext::base::SyntaxExtension;
use syntax::symbol::Symbol;
use syntax_pos;

pub use rustc::middle::cstore::{NativeLibrary, NativeLibraryKind, LinkagePreference};
pub use rustc::middle::cstore::{NativeStatic, NativeFramework, NativeUnknown};
pub use rustc::middle::cstore::{CrateSource, LinkMeta, LibSource};

// A map from external crate numbers (as decoded from some crate file) to
// local crate numbers (as generated during this session). Each external
// crate may refer to types in other external crates, and each has their
// own crate numbers.
pub type CrateNumMap = IndexVec<CrateNum, CrateNum>;

pub enum MetadataBlob {
    Inflated(Bytes),
    Archive(locator::ArchiveMetadata),
    Raw(Vec<u8>),
}

/// Holds information about a syntax_pos::FileMap imported from another crate.
/// See `imported_filemaps()` for more information.
pub struct ImportedFileMap {
    /// This FileMap's byte-offset within the codemap of its original crate
    pub original_start_pos: syntax_pos::BytePos,
    /// The end of this FileMap within the codemap of its original crate
    pub original_end_pos: syntax_pos::BytePos,
    /// The imported FileMap's representation within the local codemap
    pub translated_filemap: Rc<syntax_pos::FileMap>,
}

pub struct CrateMetadata {
    pub name: Symbol,

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
    pub def_path_table: DefPathTable,

    pub exported_symbols: FxHashSet<DefIndex>,

    pub dep_kind: Cell<DepKind>,
    pub source: CrateSource,

    pub proc_macros: Option<Vec<(ast::Name, Rc<SyntaxExtension>)>>,
    // Foreign items imported from a dylib (Windows only)
    pub dllimport_foreign_items: FxHashSet<DefIndex>,
}

pub struct CStore {
    pub dep_graph: DepGraph,
    metas: RefCell<FxHashMap<CrateNum, Rc<CrateMetadata>>>,
    /// Map from NodeId's of local extern crate statements to crate numbers
    extern_mod_crate_map: RefCell<NodeMap<CrateNum>>,
    used_libraries: RefCell<Vec<NativeLibrary>>,
    used_link_args: RefCell<Vec<String>>,
    statically_included_foreign_items: RefCell<FxHashSet<DefIndex>>,
    pub dllimport_foreign_items: RefCell<FxHashSet<DefIndex>>,
    pub visible_parent_map: RefCell<DefIdMap<DefId>>,
}

impl CStore {
    pub fn new(dep_graph: &DepGraph) -> CStore {
        CStore {
            dep_graph: dep_graph.clone(),
            metas: RefCell::new(FxHashMap()),
            extern_mod_crate_map: RefCell::new(FxHashMap()),
            used_libraries: RefCell::new(Vec::new()),
            used_link_args: RefCell::new(Vec::new()),
            statically_included_foreign_items: RefCell::new(FxHashSet()),
            dllimport_foreign_items: RefCell::new(FxHashSet()),
            visible_parent_map: RefCell::new(FxHashMap()),
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

    pub fn iter_crate_data<I>(&self, mut i: I)
        where I: FnMut(CrateNum, &Rc<CrateMetadata>)
    {
        for (&k, v) in self.metas.borrow().iter() {
            i(k, v);
        }
    }

    pub fn reset(&self) {
        self.metas.borrow_mut().clear();
        self.extern_mod_crate_map.borrow_mut().clear();
        self.used_libraries.borrow_mut().clear();
        self.used_link_args.borrow_mut().clear();
        self.statically_included_foreign_items.borrow_mut().clear();
    }

    pub fn crate_dependencies_in_rpo(&self, krate: CrateNum) -> Vec<CrateNum> {
        let mut ordering = Vec::new();
        self.push_dependencies_in_postorder(&mut ordering, krate);
        ordering.reverse();
        ordering
    }

    pub fn push_dependencies_in_postorder(&self, ordering: &mut Vec<CrateNum>, krate: CrateNum) {
        if ordering.contains(&krate) {
            return;
        }

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
    pub fn do_get_used_crates(&self,
                              prefer: LinkagePreference)
                              -> Vec<(CrateNum, LibSource)> {
        let mut ordering = Vec::new();
        for (&num, _) in self.metas.borrow().iter() {
            self.push_dependencies_in_postorder(&mut ordering, num);
        }
        info!("topological ordering: {:?}", ordering);
        ordering.reverse();
        let mut libs = self.metas
            .borrow()
            .iter()
            .filter_map(|(&cnum, data)| {
                if data.dep_kind.get().macros_only() { return None; }
                let path = match prefer {
                    LinkagePreference::RequireDynamic => data.source.dylib.clone().map(|p| p.0),
                    LinkagePreference::RequireStatic => data.source.rlib.clone().map(|p| p.0),
                };
                let path = match path {
                    Some(p) => LibSource::Some(p),
                    None => {
                        if data.source.rmeta.is_some() {
                            LibSource::MetadataOnly
                        } else {
                            LibSource::None
                        }
                    }
                };
                Some((cnum, path))
            })
            .collect::<Vec<_>>();
        libs.sort_by(|&(a, _), &(b, _)| {
            let a = ordering.iter().position(|x| *x == a);
            let b = ordering.iter().position(|x| *x == b);
            a.cmp(&b)
        });
        libs
    }

    pub fn add_used_library(&self, lib: NativeLibrary) {
        assert!(!lib.name.as_str().is_empty());
        self.used_libraries.borrow_mut().push(lib);
    }

    pub fn get_used_libraries(&self) -> &RefCell<Vec<NativeLibrary>> {
        &self.used_libraries
    }

    pub fn add_used_link_args(&self, args: &str) {
        for s in args.split(' ').filter(|s| !s.is_empty()) {
            self.used_link_args.borrow_mut().push(s.to_string());
        }
    }

    pub fn get_used_link_args<'a>(&'a self) -> &'a RefCell<Vec<String>> {
        &self.used_link_args
    }

    pub fn add_extern_mod_stmt_cnum(&self, emod_id: ast::NodeId, cnum: CrateNum) {
        self.extern_mod_crate_map.borrow_mut().insert(emod_id, cnum);
    }

    pub fn add_statically_included_foreign_item(&self, id: DefIndex) {
        self.statically_included_foreign_items.borrow_mut().insert(id);
    }

    pub fn do_is_statically_included_foreign_item(&self, def_id: DefId) -> bool {
        assert!(def_id.krate == LOCAL_CRATE);
        self.statically_included_foreign_items.borrow().contains(&def_id.index)
    }

    pub fn do_extern_mod_stmt_cnum(&self, emod_id: ast::NodeId) -> Option<CrateNum> {
        self.extern_mod_crate_map.borrow().get(&emod_id).cloned()
    }
}

impl CrateMetadata {
    pub fn name(&self) -> Symbol {
        self.root.name
    }
    pub fn hash(&self) -> Svh {
        self.root.hash
    }
    pub fn disambiguator(&self) -> Symbol {
        self.root.disambiguator
    }

    pub fn is_staged_api(&self) -> bool {
        self.get_item_attrs(CRATE_DEF_INDEX)
            .iter()
            .any(|attr| attr.name() == "stable" || attr.name() == "unstable")
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
