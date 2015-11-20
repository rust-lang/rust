// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

// The crate store - a central repo for information collected about external
// crates and libraries

pub use self::MetadataBlob::*;
pub use self::LinkagePreference::*;
pub use self::NativeLibraryKind::*;

use back::svh::Svh;
use metadata::{creader, decoder, index, loader};
use session::search_paths::PathKind;
use util::nodemap::{FnvHashMap, NodeMap, NodeSet};

use std::cell::{RefCell, Ref, Cell};
use std::rc::Rc;
use std::path::PathBuf;
use flate::Bytes;
use syntax::ast;
use syntax::attr;
use syntax::codemap;
use syntax::parse::token;
use syntax::parse::token::IdentInterner;
use syntax::util::small_vector::SmallVector;
use front::map as ast_map;

// A map from external crate numbers (as decoded from some crate file) to
// local crate numbers (as generated during this session). Each external
// crate may refer to types in other external crates, and each has their
// own crate numbers.
pub type cnum_map = FnvHashMap<ast::CrateNum, ast::CrateNum>;

pub enum MetadataBlob {
    MetadataVec(Bytes),
    MetadataArchive(loader::ArchiveMetadata),
}

/// Holds information about a codemap::FileMap imported from another crate.
/// See creader::import_codemap() for more information.
pub struct ImportedFileMap {
    /// This FileMap's byte-offset within the codemap of its original crate
    pub original_start_pos: codemap::BytePos,
    /// The end of this FileMap within the codemap of its original crate
    pub original_end_pos: codemap::BytePos,
    /// The imported FileMap's representation within the local codemap
    pub translated_filemap: Rc<codemap::FileMap>
}

pub struct crate_metadata {
    pub name: String,
    pub local_path: RefCell<SmallVector<ast_map::PathElem>>,
    pub local_def_path: RefCell<ast_map::DefPath>,
    pub data: MetadataBlob,
    pub cnum_map: RefCell<cnum_map>,
    pub cnum: ast::CrateNum,
    pub codemap_import_info: RefCell<Vec<ImportedFileMap>>,
    pub span: codemap::Span,
    pub staged_api: bool,

    pub index: index::Index,
    pub xref_index: index::DenseIndex,

    /// Flag if this crate is required by an rlib version of this crate, or in
    /// other words whether it was explicitly linked to. An example of a crate
    /// where this is false is when an allocator crate is injected into the
    /// dependency list, and therefore isn't actually needed to link an rlib.
    pub explicitly_linked: Cell<bool>,
}

#[derive(Copy, Debug, PartialEq, Clone)]
pub enum LinkagePreference {
    RequireDynamic,
    RequireStatic,
}

enum_from_u32! {
    #[derive(Copy, Clone, PartialEq)]
    pub enum NativeLibraryKind {
        NativeStatic,    // native static library (.a archive)
        NativeFramework, // OSX-specific
        NativeUnknown,   // default way to specify a dynamic library
    }
}

// Where a crate came from on the local filesystem. One of these two options
// must be non-None.
#[derive(PartialEq, Clone)]
pub struct CrateSource {
    pub dylib: Option<(PathBuf, PathKind)>,
    pub rlib: Option<(PathBuf, PathKind)>,
    pub cnum: ast::CrateNum,
}

pub struct CStore {
    metas: RefCell<FnvHashMap<ast::CrateNum, Rc<crate_metadata>>>,
    /// Map from NodeId's of local extern crate statements to crate numbers
    extern_mod_crate_map: RefCell<NodeMap<ast::CrateNum>>,
    used_crate_sources: RefCell<Vec<CrateSource>>,
    used_libraries: RefCell<Vec<(String, NativeLibraryKind)>>,
    used_link_args: RefCell<Vec<String>>,
    statically_included_foreign_items: RefCell<NodeSet>,
    pub intr: Rc<IdentInterner>,
}

/// Item definitions in the currently-compiled crate would have the CrateNum
/// LOCAL_CRATE in their DefId.
pub const LOCAL_CRATE: ast::CrateNum = 0;

impl CStore {
    pub fn new(intr: Rc<IdentInterner>) -> CStore {
        CStore {
            metas: RefCell::new(FnvHashMap()),
            extern_mod_crate_map: RefCell::new(FnvHashMap()),
            used_crate_sources: RefCell::new(Vec::new()),
            used_libraries: RefCell::new(Vec::new()),
            used_link_args: RefCell::new(Vec::new()),
            intr: intr,
            statically_included_foreign_items: RefCell::new(NodeSet()),
        }
    }

    pub fn next_crate_num(&self) -> ast::CrateNum {
        self.metas.borrow().len() as ast::CrateNum + 1
    }

    pub fn get_crate_data(&self, cnum: ast::CrateNum) -> Rc<crate_metadata> {
        self.metas.borrow().get(&cnum).unwrap().clone()
    }

    pub fn get_crate_hash(&self, cnum: ast::CrateNum) -> Svh {
        let cdata = self.get_crate_data(cnum);
        decoder::get_crate_hash(cdata.data())
    }

    pub fn set_crate_data(&self, cnum: ast::CrateNum, data: Rc<crate_metadata>) {
        self.metas.borrow_mut().insert(cnum, data);
    }

    pub fn iter_crate_data<I>(&self, mut i: I) where
        I: FnMut(ast::CrateNum, &Rc<crate_metadata>),
    {
        for (&k, v) in self.metas.borrow().iter() {
            i(k, v);
        }
    }

    /// Like `iter_crate_data`, but passes source paths (if available) as well.
    pub fn iter_crate_data_origins<I>(&self, mut i: I) where
        I: FnMut(ast::CrateNum, &crate_metadata, Option<CrateSource>),
    {
        for (&k, v) in self.metas.borrow().iter() {
            let origin = self.do_get_used_crate_source(k);
            origin.as_ref().map(|cs| { assert!(k == cs.cnum); });
            i(k, &**v, origin);
        }
    }

    pub fn add_used_crate_source(&self, src: CrateSource) {
        let mut used_crate_sources = self.used_crate_sources.borrow_mut();
        if !used_crate_sources.contains(&src) {
            used_crate_sources.push(src);
        }
    }

    // TODO: killdo
    pub fn do_get_used_crate_source(&self, cnum: ast::CrateNum)
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

    // This method is used when generating the command line to pass through to
    // system linker. The linker expects undefined symbols on the left of the
    // command line to be defined in libraries on the right, not the other way
    // around. For more info, see some comments in the add_used_library function
    // below.
    //
    // In order to get this left-to-right dependency ordering, we perform a
    // topological sort of all crates putting the leaves at the right-most
    // positions.
    // TODO: killdo
    pub fn do_get_used_crates(&self, prefer: LinkagePreference)
                              -> Vec<(ast::CrateNum, Option<PathBuf>)> {
        let mut ordering = Vec::new();
        fn visit(cstore: &CStore, cnum: ast::CrateNum,
                 ordering: &mut Vec<ast::CrateNum>) {
            if ordering.contains(&cnum) { return }
            let meta = cstore.get_crate_data(cnum);
            for (_, &dep) in meta.cnum_map.borrow().iter() {
                visit(cstore, dep, ordering);
            }
            ordering.push(cnum);
        }
        for (&num, _) in self.metas.borrow().iter() {
            visit(self, num, &mut ordering);
        }
        info!("topological ordering: {:?}", ordering);
        ordering.reverse();
        let mut libs = self.used_crate_sources.borrow()
            .iter()
            .map(|src| (src.cnum, match prefer {
                RequireDynamic => src.dylib.clone().map(|p| p.0),
                RequireStatic => src.rlib.clone().map(|p| p.0),
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
                                    cnum: ast::CrateNum) {
        self.extern_mod_crate_map.borrow_mut().insert(emod_id, cnum);
    }

    pub fn find_extern_mod_stmt_cnum(&self, emod_id: ast::NodeId)
                                     -> Option<ast::CrateNum> {
        self.extern_mod_crate_map.borrow().get(&emod_id).cloned()
    }

    pub fn add_statically_included_foreign_item(&self, id: ast::NodeId) {
        self.statically_included_foreign_items.borrow_mut().insert(id);
    }

    pub fn is_statically_included_foreign_item(&self, id: ast::NodeId) -> bool {
        self.statically_included_foreign_items.borrow().contains(&id)
    }
}

impl crate_metadata {
    pub fn data<'a>(&'a self) -> &'a [u8] { self.data.as_slice() }
    pub fn name(&self) -> String { decoder::get_crate_name(self.data()) }
    pub fn hash(&self) -> Svh { decoder::get_crate_hash(self.data()) }
    pub fn imported_filemaps<'a>(&'a self, codemap: &codemap::CodeMap)
                                 -> Ref<'a, Vec<ImportedFileMap>> {
        let filemaps = self.codemap_import_info.borrow();
        if filemaps.is_empty() {
            drop(filemaps);
            let filemaps = creader::import_codemap(codemap, &self.data);

            // This shouldn't borrow twice, but there is no way to downgrade RefMut to Ref.
            *self.codemap_import_info.borrow_mut() = filemaps;
            self.codemap_import_info.borrow()
        } else {
            filemaps
        }
    }

    pub fn with_local_path<T, F>(&self, f: F) -> T
        where F: Fn(&[ast_map::PathElem]) -> T
    {
        let cpath = self.local_path.borrow();
        if cpath.is_empty() {
            let name = ast_map::PathMod(token::intern(&self.name));
            f(&[name])
        } else {
            f(cpath.as_slice())
        }
    }

    pub fn update_local_path<'a, 'b>(&self, candidate: ast_map::PathElems<'a, 'b>) {
        let mut cpath = self.local_path.borrow_mut();
        let cap = cpath.len();
        match cap {
            0 => *cpath = candidate.collect(),
            1 => (),
            _ => {
                let candidate: SmallVector<_> = candidate.collect();
                if candidate.len() < cap {
                    *cpath = candidate;
                }
            },
        }
    }

    pub fn local_def_path(&self) -> ast_map::DefPath {
        let local_def_path = self.local_def_path.borrow();
        if local_def_path.is_empty() {
            let name = ast_map::DefPathData::DetachedCrate(token::intern(&self.name));
            vec![ast_map::DisambiguatedDefPathData { data: name, disambiguator: 0 }]
        } else {
            local_def_path.clone()
        }
    }

    pub fn update_local_def_path(&self, candidate: ast_map::DefPath) {
        let mut local_def_path = self.local_def_path.borrow_mut();
        if local_def_path.is_empty() || candidate.len() < local_def_path.len() {
            *local_def_path = candidate;
        }
    }

    pub fn is_allocator(&self) -> bool {
        let attrs = decoder::get_crate_attributes(self.data());
        attr::contains_name(&attrs, "allocator")
    }

    pub fn needs_allocator(&self) -> bool {
        let attrs = decoder::get_crate_attributes(self.data());
        attr::contains_name(&attrs, "needs_allocator")
    }
}

impl MetadataBlob {
    pub fn as_slice<'a>(&'a self) -> &'a [u8] {
        let slice = match *self {
            MetadataVec(ref vec) => &vec[..],
            MetadataArchive(ref ar) => ar.as_slice(),
        };
        if slice.len() < 4 {
            &[] // corrupt metadata
        } else {
            let len = (((slice[0] as u32) << 24) |
                       ((slice[1] as u32) << 16) |
                       ((slice[2] as u32) << 8) |
                       ((slice[3] as u32) << 0)) as usize;
            if len + 4 <= slice.len() {
                &slice[4.. len + 4]
            } else {
                &[] // corrupt or old metadata
            }
        }
    }
}
