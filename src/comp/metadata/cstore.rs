// The crate store - a central repo for information collected about external
// crates and libraries

import std::map;
import std::vec;
import std::str;
import syntax::ast;

export cstore;
export cnum_map;
export crate_metadata;
export mk_cstore;
export get_crate_data;
export set_crate_data;
export have_crate_data;
export iter_crate_data;
export add_used_crate_file;
export get_used_crate_files;
export add_used_library;
export get_used_libraries;
export add_used_link_args;
export get_used_link_args;
export add_use_stmt_cnum;
export get_use_stmt_cnum;

// A map from external crate numbers (as decoded from some crate file) to
// local crate numbers (as generated during this session). Each external
// crate may refer to types in other external crates, and each has their
// own crate numbers.
type cnum_map = map::hashmap[ast::crate_num, ast::crate_num];

type crate_metadata = rec(str name, @u8[] data, cnum_map cnum_map);

// This is a bit of an experiment at encapsulating the data in cstore. By
// keeping all the data in a non-exported tag variant, it's impossible for
// other modules to access the cstore's private data. This could also be
// achieved with an obj, but at the expense of a vtable. Not sure if this is a
// good pattern or not.
tag cstore {
    private(cstore_private);
}

type cstore_private = @rec(map::hashmap[ast::crate_num, crate_metadata] metas,
                           use_crate_map use_crate_map,
                           mutable vec[str] used_crate_files,
                           mutable vec[str] used_libraries,
                           mutable vec[str] used_link_args);

// Map from node_id's of local use statements to crate numbers
type use_crate_map = map::hashmap[ast::node_id, ast::crate_num];

// Internal method to retrieve the data from the cstore
fn p(&cstore cstore) -> cstore_private {
    alt (cstore) {
        case (private(?p)) { p }
    }
}

fn mk_cstore() -> cstore {
    auto meta_cache = map::new_int_hash[crate_metadata]();
    auto crate_map = map::new_int_hash[ast::crate_num]();
    ret private(@rec(metas = meta_cache,
                     use_crate_map = crate_map,
                     mutable used_crate_files = [],
                     mutable used_libraries = [],
                     mutable used_link_args = []));
}

fn get_crate_data(&cstore cstore, ast::crate_num cnum) -> crate_metadata {
    ret p(cstore).metas.get(cnum);
}

fn set_crate_data(&cstore cstore, ast::crate_num cnum, &crate_metadata data) {
    p(cstore).metas.insert(cnum, data);
}

fn have_crate_data(&cstore cstore, ast::crate_num cnum) -> bool {
    ret p(cstore).metas.contains_key(cnum);
}

iter iter_crate_data(&cstore cstore) -> @tup(ast::crate_num, crate_metadata) {
    for each (@tup(ast::crate_num, crate_metadata) kv
              in p(cstore).metas.items()) {
        put kv;
    }
}

fn add_used_crate_file(&cstore cstore, &str lib) {
    if (!vec::member(lib, p(cstore).used_crate_files)) {
        p(cstore).used_crate_files += [lib];
    }
}

fn get_used_crate_files(&cstore cstore) -> vec[str] {
    ret p(cstore).used_crate_files;
}

fn add_used_library(&cstore cstore, &str lib) -> bool {
    if (lib == "") { ret false; }

    if (vec::member(lib, p(cstore).used_libraries)) {
        ret false;
    }

    p(cstore).used_libraries += [lib];
    ret true;
}

fn get_used_libraries(&cstore cstore) -> vec[str] {
    ret p(cstore).used_libraries;
}

fn add_used_link_args(&cstore cstore, &str args) {
    p(cstore).used_link_args += str::split(args, ' ' as u8);
}

fn get_used_link_args(&cstore cstore) -> vec[str] {
    ret p(cstore).used_link_args;
}

fn add_use_stmt_cnum(&cstore cstore, ast::node_id use_id,
                     ast::crate_num cnum) {
    p(cstore).use_crate_map.insert(use_id, cnum);
}

fn get_use_stmt_cnum(&cstore cstore, ast::node_id use_id) -> ast::crate_num {
    ret p(cstore).use_crate_map.get(use_id);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
