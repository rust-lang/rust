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

use core::prelude::*;

use metadata::cstore;
use metadata::decoder;

use core::vec;
use std::oldmap;
use std;
use syntax::{ast, attr};
use syntax::parse::token::ident_interner;

// A map from external crate numbers (as decoded from some crate file) to
// local crate numbers (as generated during this session). Each external
// crate may refer to types in other external crates, and each has their
// own crate numbers.
pub type cnum_map = oldmap::HashMap<ast::crate_num, ast::crate_num>;

pub struct crate_metadata {
    name: @~str,
    data: @~[u8],
    cnum_map: cnum_map,
    cnum: ast::crate_num
}

pub struct CStore {
    priv metas: oldmap::HashMap<ast::crate_num, @crate_metadata>,
    priv extern_mod_crate_map: extern_mod_crate_map,
    priv used_crate_files: ~[Path],
    priv used_libraries: ~[~str],
    priv used_link_args: ~[~str],
    intr: @ident_interner
}

// Map from node_id's of local extern mod statements to crate numbers
type extern_mod_crate_map = oldmap::HashMap<ast::node_id, ast::crate_num>;

pub fn mk_cstore(intr: @ident_interner) -> CStore {
    let meta_cache = oldmap::HashMap();
    let crate_map = oldmap::HashMap();
    return CStore {
        metas: meta_cache,
        extern_mod_crate_map: crate_map,
        used_crate_files: ~[],
        used_libraries: ~[],
        used_link_args: ~[],
        intr: intr
    };
}

pub fn get_crate_data(cstore: @mut CStore, cnum: ast::crate_num)
                   -> @crate_metadata {
    return cstore.metas.get(&cnum);
}

pub fn get_crate_hash(cstore: @mut CStore, cnum: ast::crate_num) -> @~str {
    let cdata = get_crate_data(cstore, cnum);
    decoder::get_crate_hash(cdata.data)
}

pub fn get_crate_vers(cstore: @mut CStore, cnum: ast::crate_num) -> @~str {
    let cdata = get_crate_data(cstore, cnum);
    decoder::get_crate_vers(cdata.data)
}

pub fn set_crate_data(cstore: @mut CStore,
                      cnum: ast::crate_num,
                      data: @crate_metadata) {
    let metas = cstore.metas;
    metas.insert(cnum, data);
}

pub fn have_crate_data(cstore: @mut CStore, cnum: ast::crate_num) -> bool {
    cstore.metas.contains_key(&cnum)
}

pub fn iter_crate_data(cstore: @mut CStore,
                       i: fn(ast::crate_num, @crate_metadata)) {
    let metas = cstore.metas;
    for metas.each |&k, &v| {
        i(k, v);
    }
}

pub fn add_used_crate_file(cstore: @mut CStore, lib: &Path) {
    if !vec::contains(cstore.used_crate_files, lib) {
        cstore.used_crate_files.push(copy *lib);
    }
}

pub fn get_used_crate_files(cstore: @mut CStore) -> ~[Path] {
    return /*bad*/copy cstore.used_crate_files;
}

pub fn add_used_library(cstore: @mut CStore, lib: @~str) -> bool {
    fail_unless!(*lib != ~"");

    if cstore.used_libraries.contains(&*lib) { return false; }
    cstore.used_libraries.push(/*bad*/ copy *lib);
    true
}

pub fn get_used_libraries(cstore: @mut CStore) -> ~[~str] {
    /*bad*/copy cstore.used_libraries
}

pub fn add_used_link_args(cstore: @mut CStore, args: &str) {
    cstore.used_link_args.push_all(args.split_char(' '));
}

pub fn get_used_link_args(cstore: @mut CStore) -> ~[~str] {
    /*bad*/copy cstore.used_link_args
}

pub fn add_extern_mod_stmt_cnum(cstore: @mut CStore,
                                emod_id: ast::node_id,
                                cnum: ast::crate_num) {
    let extern_mod_crate_map = cstore.extern_mod_crate_map;
    extern_mod_crate_map.insert(emod_id, cnum);
}

pub fn find_extern_mod_stmt_cnum(cstore: @mut CStore,
                                 emod_id: ast::node_id)
                       -> Option<ast::crate_num> {
    let extern_mod_crate_map = cstore.extern_mod_crate_map;
    extern_mod_crate_map.find(&emod_id)
}

// returns hashes of crates directly used by this crate. Hashes are
// sorted by crate name.
pub fn get_dep_hashes(cstore: @mut CStore) -> ~[~str] {
    struct crate_hash { name: @~str, hash: @~str }
    let mut result = ~[];

    let extern_mod_crate_map = cstore.extern_mod_crate_map;
    for extern_mod_crate_map.each_value |&cnum| {
        let cdata = cstore::get_crate_data(cstore, cnum);
        let hash = decoder::get_crate_hash(cdata.data);
        debug!("Add hash[%s]: %s", *cdata.name, *hash);
        result.push(crate_hash {
            name: cdata.name,
            hash: hash
        });
    }

    let sorted = std::sort::merge_sort(result, |a, b| a.name <= b.name);

    debug!("sorted:");
    for sorted.each |x| {
        debug!("  hash[%s]: %s", *x.name, *x.hash);
    }

    sorted.map(|ch| /*bad*/copy *ch.hash)
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
