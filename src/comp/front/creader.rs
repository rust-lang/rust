// -*- rust -*-

import driver.session;
import front.ast;
import lib.llvm.llvmext;
import lib.llvm.mk_object_file;
import lib.llvm.mk_section_iter;
import middle.fold;
import util.common;
import util.common.span;

import std._str;
import std.fs;
import std.os;
import std.map.hashmap;

// TODO: map to a real type here.
type env = @rec(
    @hashmap[str, @ast.external_crate_info] crate_cache,
    vec[str] library_search_paths
);

// TODO: return something
fn load_crate(ast.ident ident, vec[str] library_search_paths) -> @() {
    for (str library_search_path in library_search_paths) {
        auto path = fs.connect(library_search_path, ident);
        // TODO
    }

    ret @();
}

fn fold_view_item_use(&env e, &span sp, ast.ident ident,
        vec[@ast.meta_item] meta_items, ast.def_id id, ast.ann orig_ann)
        -> @ast.view_item {
    auto external_crate;
    if (!e.crate_cache.contains_key(ident)) {
        external_crate = load_crate(ident, e.library_search_paths);
        e.crate_cache.insert(ident, external_crate);
    } else {
        external_crate = e.crate_cache.get(ident);
    }

    auto ann = ast.ann_crate(external_crate);
    auto viu = ast.view_item_use(ident, meta_items, id, ann);
    ret @fold.respan[ast.view_item_](sp, viu);
}

// Reads external crates referenced by "use" directives.
fn read_crates(session.session sess,
               @ast.crate crate,
               vec[str] library_search_paths) -> @ast.crate {
    auto e = @rec(
        crate_cache=@common.new_str_hash[@ast.external_crate_info](),
        library_search_paths=library_search_paths
    );

    auto f = fold_view_item_use;
    auto fld = @rec(fold_view_item_use=f with *fold.new_identity_fold[env]());
    ret fold.fold_crate[env](e, fld, crate);
}

