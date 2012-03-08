// Searching for information from the cstore

import syntax::ast;
import syntax::ast_util;
import middle::{ty, ast_map};
import option::{some, none};
import driver::session;
import middle::trans::common::maps;
import std::map::hashmap;

export get_symbol;
export get_type_param_count;
export lookup_defs;
export lookup_method_purity;
export get_enum_variants;
export get_impls_for_mod;
export get_iface_methods;
export get_type;
export get_impl_iface;
export get_item_path;
export maybe_get_item_ast;

fn get_symbol(cstore: cstore::cstore, def: ast::def_id) -> str {
    let cdata = cstore::get_crate_data(cstore, def.crate).data;
    ret decoder::get_symbol(cdata, def.node);
}

fn get_type_param_count(cstore: cstore::cstore, def: ast::def_id) -> uint {
    let cdata = cstore::get_crate_data(cstore, def.crate).data;
    ret decoder::get_type_param_count(cdata, def.node);
}

fn lookup_defs(cstore: cstore::cstore, cnum: ast::crate_num,
               path: [ast::ident]) -> [ast::def] {
    let result = [];
    for (c, data, def) in resolve_path(cstore, cnum, path) {
        result += [decoder::lookup_def(c, data, def)];
    }
    ret result;
}

fn lookup_method_purity(cstore: cstore::cstore, did: ast::def_id)
    -> ast::purity {
    let cdata = cstore::get_crate_data(cstore, did.crate).data;
    alt check decoder::lookup_def(did.crate, cdata, did) {
      ast::def_fn(_, p) { p }
    }
}

fn resolve_path(cstore: cstore::cstore, cnum: ast::crate_num,
                path: [ast::ident]) ->
    [(ast::crate_num, @[u8], ast::def_id)] {
    let cm = cstore::get_crate_data(cstore, cnum);
    #debug("resolve_path %s in crates[%d]:%s",
           str::connect(path, "::"), cnum, cm.name);
    let result = [];
    for def in decoder::resolve_path(path, cm.data) {
        if def.crate == ast::local_crate {
            result += [(cnum, cm.data, def)];
        } else {
            if cm.cnum_map.contains_key(def.crate) {
                // This reexport is itself a reexport from anther crate
                let next_cnum = cm.cnum_map.get(def.crate);
                let next_cm_data = cstore::get_crate_data(cstore, next_cnum);
                result += [(next_cnum, next_cm_data.data, def)];
            }
        }
    }
    ret result;
}

fn get_item_path(tcx: ty::ctxt, def: ast::def_id) -> ast_map::path {
    let cstore = tcx.sess.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    let path = decoder::get_item_path(cdata, def.node);

    // FIXME #1920: This path is not always correct if the crate is not linked
    // into the root namespace.
    [ast_map::path_mod(cdata.name)] + path
}

// Finds the AST for this item in the crate metadata, if any.  If the item was
// not marked for inlining, then the AST will not be present and hence none
// will be returned.
fn maybe_get_item_ast(tcx: ty::ctxt, maps: maps, def: ast::def_id)
    -> option<ast::inlined_item> {
    let cstore = tcx.sess.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::maybe_get_item_ast(cdata, tcx, maps, def.node)
}

fn get_enum_variants(tcx: ty::ctxt, def: ast::def_id) -> [ty::variant_info] {
    let cstore = tcx.sess.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    ret decoder::get_enum_variants(cdata, def.node, tcx)
}

fn get_impls_for_mod(cstore: cstore::cstore, def: ast::def_id,
                     name: option<ast::ident>)
    -> @[@middle::resolve::_impl] {
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::get_impls_for_mod(cdata, def.node, name)
}

fn get_iface_methods(tcx: ty::ctxt, def: ast::def_id) -> @[ty::method] {
    let cstore = tcx.sess.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::get_iface_methods(cdata, def.node, tcx)
}

fn get_type(tcx: ty::ctxt, def: ast::def_id) -> ty::ty_param_bounds_and_ty {
    let cstore = tcx.sess.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::get_type(cdata, def.node, tcx)
}

fn get_impl_iface(tcx: ty::ctxt, def: ast::def_id)
    -> option<ty::t> {
    let cstore = tcx.sess.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::get_impl_iface(cdata, def.node, tcx)
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
