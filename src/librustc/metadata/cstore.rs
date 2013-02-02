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

use metadata::creader;
use metadata::cstore;
use metadata::decoder;

use core::option;
use core::str;
use core::vec;
use std::oldmap::HashMap;
use std::oldmap;
use std;
use syntax::{ast, attr};
use syntax::parse::token::ident_interner;

// A map from external crate numbers (as decoded from some crate file) to
// local crate numbers (as generated during this session). Each external
// crate may refer to types in other external crates, and each has their
// own crate numbers.
pub type cnum_map = oldmap::HashMap<ast::crate_num, ast::crate_num>;

pub type crate_metadata = @{name: ~str,
                            data: @~[u8],
                            cnum_map: cnum_map,
                            cnum: ast::crate_num};

// This is a bit of an experiment at encapsulating the data in cstore. By
// keeping all the data in a non-exported enum variant, it's impossible for
// other modules to access the cstore's private data. This could also be
// achieved with an obj, but at the expense of a vtable. Not sure if this is a
// good pattern or not.
pub enum CStore { private(cstore_private), }

type cstore_private =
    @{metas: oldmap::HashMap<ast::crate_num, crate_metadata>,
      use_crate_map: use_crate_map,
      mut used_crate_files: ~[Path],
      mut used_libraries: ~[~str],
      mut used_link_args: ~[~str],
      intr: @ident_interner};

// Map from node_id's of local use statements to crate numbers
type use_crate_map = oldmap::HashMap<ast::node_id, ast::crate_num>;

// Internal method to retrieve the data from the cstore
pure fn p(cstore: CStore) -> cstore_private {
    match cstore { private(p) => p }
}

pub fn mk_cstore(intr: @ident_interner) -> CStore {
    let meta_cache = oldmap::HashMap();
    let crate_map = oldmap::HashMap();
    return private(@{metas: meta_cache,
                     use_crate_map: crate_map,
                     mut used_crate_files: ~[],
                     mut used_libraries: ~[],
                     mut used_link_args: ~[],
                     intr: intr});
}

pub fn get_crate_data(cstore: CStore, cnum: ast::crate_num)
                   -> crate_metadata {
    return p(cstore).metas.get(cnum);
}

pub fn get_crate_hash(cstore: CStore, cnum: ast::crate_num) -> ~str {
    let cdata = get_crate_data(cstore, cnum);
    return decoder::get_crate_hash(cdata.data);
}

pub fn get_crate_vers(cstore: CStore, cnum: ast::crate_num) -> ~str {
    let cdata = get_crate_data(cstore, cnum);
    return decoder::get_crate_vers(cdata.data);
}

pub fn set_crate_data(cstore: CStore,
                      cnum: ast::crate_num,
                      data: crate_metadata) {
    p(cstore).metas.insert(cnum, data);
}

pub fn have_crate_data(cstore: CStore, cnum: ast::crate_num) -> bool {
    return p(cstore).metas.contains_key_ref(&cnum);
}

pub fn iter_crate_data(cstore: CStore,
                       i: fn(ast::crate_num, crate_metadata)) {
    for p(cstore).metas.each_ref |&k, &v| { i(k, v);};
}

pub fn add_used_crate_file(cstore: CStore, lib: &Path) {
    if !vec::contains(p(cstore).used_crate_files, lib) {
        p(cstore).used_crate_files.push(copy *lib);
    }
}

pub fn get_used_crate_files(cstore: CStore) -> ~[Path] {
    return /*bad*/copy p(cstore).used_crate_files;
}

pub fn add_used_library(cstore: CStore, +lib: ~str) -> bool {
    assert lib != ~"";

    if vec::contains(p(cstore).used_libraries, &lib) { return false; }
    p(cstore).used_libraries.push(lib);
    return true;
}

pub fn get_used_libraries(cstore: CStore) -> ~[~str] {
    return /*bad*/copy p(cstore).used_libraries;
}

pub fn add_used_link_args(cstore: CStore, args: ~str) {
    p(cstore).used_link_args.push_all(str::split_char(args, ' '));
}

pub fn get_used_link_args(cstore: CStore) -> ~[~str] {
    return /*bad*/copy p(cstore).used_link_args;
}

pub fn add_use_stmt_cnum(cstore: CStore, use_id: ast::node_id,
                         cnum: ast::crate_num) {
    p(cstore).use_crate_map.insert(use_id, cnum);
}

pub fn find_use_stmt_cnum(cstore: CStore,
                          use_id: ast::node_id) -> Option<ast::crate_num> {
    p(cstore).use_crate_map.find(use_id)
}

// returns hashes of crates directly used by this crate. Hashes are
// sorted by crate name.
pub fn get_dep_hashes(cstore: CStore) -> ~[~str] {
    type crate_hash = {name: ~str, hash: ~str};
    let mut result = ~[];

    for p(cstore).use_crate_map.each_value_ref |&cnum| {
        let cdata = cstore::get_crate_data(cstore, cnum);
        let hash = decoder::get_crate_hash(cdata.data);
        debug!("Add hash[%s]: %s", cdata.name, hash);
        result.push({name: /*bad*/copy cdata.name, hash: hash});
    };
    pure fn lteq(a: &crate_hash, b: &crate_hash) -> bool {a.name <= b.name}
    let sorted = std::sort::merge_sort(result, lteq);
    debug!("sorted:");
    for sorted.each |x| {
        debug!("  hash[%s]: %s", x.name, x.hash);
    }
    fn mapper(ch: &crate_hash) -> ~str { return /*bad*/copy ch.hash; }
    return vec::map(sorted, mapper);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
