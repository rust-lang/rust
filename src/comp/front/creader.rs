// -*- rust -*-

import driver.session;
import front.ast;
import middle.fold;
import util.common;
import util.common.span;
import std.map.hashmap;

// TODO: map to a real type here.
type env = @hashmap[str, @ast.external_crate_info];

fn fold_view_item_use(&env e, &span sp, ast.ident ident,
        vec[@ast.meta_item] meta_items, ast.def_id id) -> @ast.view_item {
    // TODO: find the crate

    auto viu = ast.view_item_use(ident, meta_items, id);
    ret @fold.respan[ast.view_item_](sp, viu);
}

// Reads external crates referenced by "use" directives.
fn read_crates(session.session sess, @ast.crate crate) -> @ast.crate {
    auto external_crates = @common.new_str_hash[@ast.external_crate_info]();
    auto f = fold_view_item_use;
    auto fld = @rec(fold_view_item_use=f with *fold.new_identity_fold[env]());
    ret fold.fold_crate[env](external_crates, fld, crate);
}

