// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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


use metadata::cstore;
use metadata::decoder;
use metadata::loader;

use std::hashmap::HashMap;
use syntax::ast;
use syntax::parse::token::ident_interner;

// A map from external crate numbers (as decoded from some crate file) to
// local crate numbers (as generated during this session). Each external
// crate may refer to types in other external crates, and each has their
// own crate numbers.
pub type cnum_map = @mut HashMap<ast::CrateNum, ast::CrateNum>;

pub enum MetadataBlob {
    MetadataVec(~[u8]),
    MetadataArchive(loader::ArchiveMetadata),
}

pub struct crate_metadata {
    name: @str,
    data: MetadataBlob,
    cnum_map: cnum_map,
    cnum: ast::CrateNum
}

#[deriving(Eq)]
pub enum LinkagePreference {
    RequireDynamic,
    RequireStatic,
}

#[deriving(Eq, FromPrimitive)]
pub enum NativeLibaryKind {
    NativeStatic,    // native static library (.a archive)
    NativeFramework, // OSX-specific
    NativeUnknown,   // default way to specify a dynamic library
}

// Where a crate came from on the local filesystem. One of these two options
// must be non-None.
#[deriving(Eq)]
pub struct CrateSource {
    dylib: Option<Path>,
    rlib: Option<Path>,
    cnum: ast::CrateNum,
}

pub struct CStore {
    priv metas: HashMap <ast::CrateNum, @crate_metadata>,
    priv extern_mod_crate_map: extern_mod_crate_map,
    priv used_crate_sources: ~[CrateSource],
    priv used_libraries: ~[(~str, NativeLibaryKind)],
    priv used_link_args: ~[~str],
    intr: @ident_interner
}

// Map from NodeId's of local extern mod statements to crate numbers
type extern_mod_crate_map = HashMap<ast::NodeId, ast::CrateNum>;

pub fn mk_cstore(intr: @ident_interner) -> CStore {
    return CStore {
        metas: HashMap::new(),
        extern_mod_crate_map: HashMap::new(),
        used_crate_sources: ~[],
        used_libraries: ~[],
        used_link_args: ~[],
        intr: intr
    };
}

pub fn get_crate_data(cstore: &CStore, cnum: ast::CrateNum)
                   -> @crate_metadata {
    return *cstore.metas.get(&cnum);
}

pub fn get_crate_hash(cstore: &CStore, cnum: ast::CrateNum) -> @str {
    let cdata = get_crate_data(cstore, cnum);
    decoder::get_crate_hash(cdata.data())
}

pub fn get_crate_vers(cstore: &CStore, cnum: ast::CrateNum) -> @str {
    let cdata = get_crate_data(cstore, cnum);
    decoder::get_crate_vers(cdata.data())
}

pub fn set_crate_data(cstore: &mut CStore,
                      cnum: ast::CrateNum,
                      data: @crate_metadata) {
    cstore.metas.insert(cnum, data);
}

pub fn have_crate_data(cstore: &CStore, cnum: ast::CrateNum) -> bool {
    cstore.metas.contains_key(&cnum)
}

pub fn iter_crate_data(cstore: &CStore, i: |ast::CrateNum, @crate_metadata|) {
    for (&k, &v) in cstore.metas.iter() {
        i(k, v);
    }
}

pub fn add_used_crate_source(cstore: &mut CStore, src: CrateSource) {
    if !cstore.used_crate_sources.contains(&src) {
        cstore.used_crate_sources.push(src);
    }
}

pub fn get_used_crate_sources<'a>(cstore: &'a CStore) -> &'a [CrateSource] {
    cstore.used_crate_sources.as_slice()
}

pub fn get_used_crates(cstore: &CStore, prefer: LinkagePreference)
    -> ~[(ast::CrateNum, Option<Path>)]
{
    let mut ret = ~[];
    for src in cstore.used_crate_sources.iter() {
        ret.push((src.cnum, match prefer {
            RequireDynamic => src.dylib.clone(),
            RequireStatic => src.rlib.clone(),
        }));
    }
    return ret;
}

pub fn add_used_library(cstore: &mut CStore,
                        lib: ~str, kind: NativeLibaryKind) -> bool {
    assert!(!lib.is_empty());

    if cstore.used_libraries.iter().any(|&(ref x, _)| x == &lib) { return false; }
    cstore.used_libraries.push((lib, kind));
    true
}

pub fn get_used_libraries<'a>(cstore: &'a CStore) -> &'a [(~str, NativeLibaryKind)] {
    cstore.used_libraries.as_slice()
}

pub fn add_used_link_args(cstore: &mut CStore, args: &str) {
    for s in args.split(' ') {
        cstore.used_link_args.push(s.to_owned());
    }
}

pub fn get_used_link_args<'a>(cstore: &'a CStore) -> &'a [~str] {
    cstore.used_link_args.as_slice()
}

pub fn add_extern_mod_stmt_cnum(cstore: &mut CStore,
                                emod_id: ast::NodeId,
                                cnum: ast::CrateNum) {
    cstore.extern_mod_crate_map.insert(emod_id, cnum);
}

pub fn find_extern_mod_stmt_cnum(cstore: &CStore,
                                 emod_id: ast::NodeId)
                       -> Option<ast::CrateNum> {
    cstore.extern_mod_crate_map.find(&emod_id).map(|x| *x)
}

#[deriving(Clone, TotalEq, TotalOrd)]
struct crate_hash {
    name: @str,
    vers: @str,
    hash: @str,
}

// returns hashes of crates directly used by this crate. Hashes are sorted by
// (crate name, crate version, crate hash) in lexicographic order (not semver)
pub fn get_dep_hashes(cstore: &CStore) -> ~[@str] {
    let mut result = ~[];

    for (_, &cnum) in cstore.extern_mod_crate_map.iter() {
        let cdata = cstore::get_crate_data(cstore, cnum);
        let hash = decoder::get_crate_hash(cdata.data());
        let vers = decoder::get_crate_vers(cdata.data());
        debug!("Add hash[{}]: {} {}", cdata.name, vers, hash);
        result.push(crate_hash {
            name: cdata.name,
            vers: vers,
            hash: hash
        });
    }

    result.sort();

    debug!("sorted:");
    for x in result.iter() {
        debug!("  hash[{}]: {}", x.name, x.hash);
    }

    result.map(|ch| ch.hash)
}

impl crate_metadata {
    pub fn data<'a>(&'a self) -> &'a [u8] { self.data.as_slice() }
}

impl MetadataBlob {
    pub fn as_slice<'a>(&'a self) -> &'a [u8] {
        match *self {
            MetadataVec(ref vec) => vec.as_slice(),
            MetadataArchive(ref ar) => ar.as_slice(),
        }
    }
}
