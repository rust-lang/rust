// The crate store - a central repo for information collected about external
// crates and libraries

import std::map;
import std::map::hashmap;
import syntax::{ast, attr};
import syntax::ast_util::new_def_hash;

export cstore::{};
export cnum_map;
export crate_metadata;
export mk_cstore;
export get_crate_data;
export set_crate_data;
export get_crate_hash;
export get_crate_vers;
export have_crate_data;
export iter_crate_data;
export add_used_crate_file;
export get_used_crate_files;
export add_used_library;
export get_used_libraries;
export add_used_link_args;
export get_used_link_args;
export add_use_stmt_cnum;
export find_use_stmt_cnum;
export get_dep_hashes;
export get_path;


// A map from external crate numbers (as decoded from some crate file) to
// local crate numbers (as generated during this session). Each external
// crate may refer to types in other external crates, and each has their
// own crate numbers.
type cnum_map = map::hashmap<ast::crate_num, ast::crate_num>;

// Multiple items may have the same def_id in crate metadata. They may be
// renamed imports or reexports. This map keeps the "real" module path
// and def_id.
type mod_path_map = map::hashmap<ast::def_id, @str>;

type crate_metadata = @{name: str,
                        data: @[u8]/~,
                        cnum_map: cnum_map,
                        cnum: ast::crate_num};

// This is a bit of an experiment at encapsulating the data in cstore. By
// keeping all the data in a non-exported enum variant, it's impossible for
// other modules to access the cstore's private data. This could also be
// achieved with an obj, but at the expense of a vtable. Not sure if this is a
// good pattern or not.
enum cstore { private(cstore_private), }

type cstore_private =
    @{metas: map::hashmap<ast::crate_num, crate_metadata>,
      use_crate_map: use_crate_map,
      mod_path_map: mod_path_map,
      mut used_crate_files: [str]/~,
      mut used_libraries: [str]/~,
      mut used_link_args: [str]/~};

// Map from node_id's of local use statements to crate numbers
type use_crate_map = map::hashmap<ast::node_id, ast::crate_num>;

// Internal method to retrieve the data from the cstore
pure fn p(cstore: cstore) -> cstore_private {
    alt cstore { private(p) { p } }
}

fn mk_cstore() -> cstore {
    let meta_cache = map::int_hash::<crate_metadata>();
    let crate_map = map::int_hash::<ast::crate_num>();
    let mod_path_map = new_def_hash();
    ret private(@{metas: meta_cache,
                  use_crate_map: crate_map,
                  mod_path_map: mod_path_map,
                  mut used_crate_files: []/~,
                  mut used_libraries: []/~,
                  mut used_link_args: []/~});
}

fn get_crate_data(cstore: cstore, cnum: ast::crate_num) -> crate_metadata {
    ret p(cstore).metas.get(cnum);
}

fn get_crate_hash(cstore: cstore, cnum: ast::crate_num) -> @str {
    let cdata = get_crate_data(cstore, cnum);
    ret decoder::get_crate_hash(cdata.data);
}

fn get_crate_vers(cstore: cstore, cnum: ast::crate_num) -> @str {
    let cdata = get_crate_data(cstore, cnum);
    ret decoder::get_crate_vers(cdata.data);
}

fn set_crate_data(cstore: cstore, cnum: ast::crate_num,
                  data: crate_metadata) {
    p(cstore).metas.insert(cnum, data);
    vec::iter(decoder::get_crate_module_paths(data.data)) {|dp|
        let (did, path) = dp;
        let d = {crate: cnum, node: did.node};
        p(cstore).mod_path_map.insert(d, @path);
    }
}

fn have_crate_data(cstore: cstore, cnum: ast::crate_num) -> bool {
    ret p(cstore).metas.contains_key(cnum);
}

fn iter_crate_data(cstore: cstore, i: fn(ast::crate_num, crate_metadata)) {
    for p(cstore).metas.each {|k,v| i(k, v);};
}

fn add_used_crate_file(cstore: cstore, lib: str) {
    if !vec::contains(p(cstore).used_crate_files, lib) {
        vec::push(p(cstore).used_crate_files, lib);
    }
}

fn get_used_crate_files(cstore: cstore) -> [str]/~ {
    ret p(cstore).used_crate_files;
}

fn add_used_library(cstore: cstore, lib: str) -> bool {
    assert lib != "";

    if vec::contains(p(cstore).used_libraries, lib) { ret false; }
    vec::push(p(cstore).used_libraries, lib);
    ret true;
}

fn get_used_libraries(cstore: cstore) -> [str]/~ {
    ret p(cstore).used_libraries;
}

fn add_used_link_args(cstore: cstore, args: str) {
    p(cstore).used_link_args += str::split_char(args, ' ');
}

fn get_used_link_args(cstore: cstore) -> [str]/~ {
    ret p(cstore).used_link_args;
}

fn add_use_stmt_cnum(cstore: cstore, use_id: ast::node_id,
                     cnum: ast::crate_num) {
    p(cstore).use_crate_map.insert(use_id, cnum);
}

fn find_use_stmt_cnum(cstore: cstore,
                      use_id: ast::node_id) -> option<ast::crate_num> {
    p(cstore).use_crate_map.find(use_id)
}

// returns hashes of crates directly used by this crate. Hashes are
// sorted by crate name.
fn get_dep_hashes(cstore: cstore) -> [@str]/~ {
    type crate_hash = {name: @str, hash: @str};
    let mut result = []/~;

    for p(cstore).use_crate_map.each_value {|cnum|
        let cdata = cstore::get_crate_data(cstore, cnum);
        let hash = decoder::get_crate_hash(cdata.data);
        #debug("Add hash[%s]: %s", cdata.name, *hash);
        vec::push(result, {name: @cdata.name, hash: hash});
    };
    fn lteq(a: crate_hash, b: crate_hash) -> bool {
        ret *a.name <= *b.name;
    }
    let sorted = std::sort::merge_sort(lteq, result);
    #debug("sorted:");
    for sorted.each {|x|
        #debug("  hash[%s]: %s", *x.name, *x.hash);
    }
    fn mapper(ch: crate_hash) -> @str { ret ch.hash; }
    ret vec::map(sorted, mapper);
}

fn get_path(cstore: cstore, d: ast::def_id) -> [ast::ident]/~ {
    // let f = bind str::split_str(_, "::");
    option::map_default(p(cstore).mod_path_map.find(d), []/~,
                        {|ds| str::split_str(*ds, "::").map({|x|@x})})
}
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
