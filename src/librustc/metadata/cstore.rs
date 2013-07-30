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

use std::hashmap::HashMap;
use extra;
use syntax::ast;
use syntax::parse::token::ident_interner;

// A map from external crate numbers (as decoded from some crate file) to
// local crate numbers (as generated during this session). Each external
// crate may refer to types in other external crates, and each has their
// own crate numbers.
pub type cnum_map = @mut HashMap<ast::CrateNum, ast::CrateNum>;

pub struct crate_metadata {
    name: @str,
    data: @~[u8],
    cnum_map: cnum_map,
    cnum: ast::CrateNum
}

pub struct CStore {
    priv metas: HashMap <ast::CrateNum, @crate_metadata>,
    priv extern_mod_crate_map: extern_mod_crate_map,
    priv used_crate_files: ~[Path],
    priv used_libraries: ~[@str],
    priv used_link_args: ~[@str],
    intr: @ident_interner
}

// Map from NodeId's of local extern mod statements to crate numbers
type extern_mod_crate_map = HashMap<ast::NodeId, ast::CrateNum>;

pub fn mk_cstore(intr: @ident_interner) -> CStore {
    return CStore {
        metas: HashMap::new(),
        extern_mod_crate_map: HashMap::new(),
        used_crate_files: ~[],
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
    decoder::get_crate_hash(cdata.data)
}

pub fn get_crate_vers(cstore: &CStore, cnum: ast::CrateNum) -> @str {
    let cdata = get_crate_data(cstore, cnum);
    decoder::get_crate_vers(cdata.data)
}

pub fn set_crate_data(cstore: &mut CStore,
                      cnum: ast::CrateNum,
                      data: @crate_metadata) {
    cstore.metas.insert(cnum, data);
}

pub fn have_crate_data(cstore: &CStore, cnum: ast::CrateNum) -> bool {
    cstore.metas.contains_key(&cnum)
}

pub fn iter_crate_data(cstore: &CStore,
                       i: &fn(ast::CrateNum, @crate_metadata)) {
    for cstore.metas.iter().advance |(&k, &v)| {
        i(k, v);
    }
}

pub fn add_used_crate_file(cstore: &mut CStore, lib: &Path) {
    if !cstore.used_crate_files.contains(lib) {
        cstore.used_crate_files.push((*lib).clone());
    }
}

pub fn get_used_crate_files(cstore: &CStore) -> ~[Path] {
    // XXX(pcwalton): Bad copy.
    return cstore.used_crate_files.clone();
}

pub fn add_used_library(cstore: &mut CStore, lib: @str) -> bool {
    assert!(!lib.is_empty());

    if cstore.used_libraries.iter().any(|x| x == &lib) { return false; }
    cstore.used_libraries.push(lib);
    true
}

pub fn get_used_libraries<'a>(cstore: &'a CStore) -> &'a [@str] {
    let slice: &'a [@str] = cstore.used_libraries;
    slice
}

pub fn add_used_link_args(cstore: &mut CStore, args: &str) {
    for args.split_iter(' ').advance |s| {
        cstore.used_link_args.push(s.to_managed());
    }
}

pub fn get_used_link_args<'a>(cstore: &'a CStore) -> &'a [@str] {
    let slice: &'a [@str] = cstore.used_link_args;
    slice
}

pub fn add_extern_mod_stmt_cnum(cstore: &mut CStore,
                                emod_id: ast::NodeId,
                                cnum: ast::CrateNum) {
    cstore.extern_mod_crate_map.insert(emod_id, cnum);
}

pub fn find_extern_mod_stmt_cnum(cstore: &CStore,
                                 emod_id: ast::NodeId)
                       -> Option<ast::CrateNum> {
    cstore.extern_mod_crate_map.find(&emod_id).map_consume(|x| *x)
}

#[deriving(Clone)]
struct crate_hash {
    name: @str,
    vers: @str,
    hash: @str,
}

// returns hashes of crates directly used by this crate. Hashes are sorted by
// (crate name, crate version, crate hash) in lexicographic order (not semver)
pub fn get_dep_hashes(cstore: &CStore) -> ~[@str] {
    let mut result = ~[];

    for cstore.extern_mod_crate_map.each_value |&cnum| {
        let cdata = cstore::get_crate_data(cstore, cnum);
        let hash = decoder::get_crate_hash(cdata.data);
        let vers = decoder::get_crate_vers(cdata.data);
        debug!("Add hash[%s]: %s %s", cdata.name, vers, hash);
        result.push(crate_hash {
            name: cdata.name,
            vers: vers,
            hash: hash
        });
    }

    let sorted = do extra::sort::merge_sort(result) |a, b| {
        (a.name, a.vers, a.hash) <= (b.name, b.vers, b.hash)
    };

    debug!("sorted:");
    for sorted.iter().advance |x| {
        debug!("  hash[%s]: %s", x.name, x.hash);
    }

    sorted.map(|ch| ch.hash)
}
