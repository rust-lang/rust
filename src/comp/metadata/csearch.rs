// Searching for information from the cstore

import syntax::ast;
import middle::ty;
import option::{some, none};
import driver::session;

export get_symbol;
export get_type_param_count;
export lookup_defs;
export get_tag_variants;
export get_impls_for_mod;
export get_impl_methods;
export get_type;
export get_item_name;

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

fn resolve_path(cstore: cstore::cstore, cnum: ast::crate_num,
                path: [ast::ident]) ->
    [(ast::crate_num, @[u8], ast::def_id)] {
    let cm = cstore::get_crate_data(cstore, cnum);
    log #fmt("resolve_path %s in crates[%d]:%s",
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

fn get_tag_variants(tcx: ty::ctxt, def: ast::def_id) -> [ty::variant_info] {
    let cstore = tcx.sess.get_cstore();
    let cnum = def.crate;
    let cdata = cstore::get_crate_data(cstore, cnum).data;
    let resolver = bind translate_def_id(cstore, cnum, _);
    ret decoder::get_tag_variants(cdata, def, tcx, resolver)
}

fn get_impls_for_mod(cstore: cstore::cstore, def: ast::def_id,
                     name: option::t<ast::ident>)
    -> [@middle::resolve::_impl] {
    let cdata = cstore::get_crate_data(cstore, def.crate).data;
    let result = [];
    for did in decoder::get_impls_for_mod(cdata, def.node, def.crate) {
        let nm = decoder::lookup_item_name(cdata, did.node);
        if alt name { some(n) { n == nm } none. { true } } {
            result += [@{did: did,
                         ident: nm,
                         methods: decoder::lookup_impl_methods(
                             cdata, did.node, did.crate)}];
        }
    }
    result
}

fn get_impl_methods(cstore: cstore::cstore, def: ast::def_id)
    -> [@middle::resolve::method_info] {
    let cdata = cstore::get_crate_data(cstore, def.crate).data;
    decoder::lookup_impl_methods(cdata, def.node, def.crate)
}

fn get_type(tcx: ty::ctxt, def: ast::def_id) -> ty::ty_param_kinds_and_ty {
    let cstore = tcx.sess.get_cstore();
    let cnum = def.crate;
    let cdata = cstore::get_crate_data(cstore, cnum).data;
    let resolver = bind translate_def_id(cstore, cnum, _);
    decoder::get_type(cdata, def, tcx, resolver)
}

fn get_item_name(cstore: cstore::cstore, cnum: int, id: int) -> ast::ident {
    let cdata = cstore::get_crate_data(cstore, cnum).data;
    ret decoder::lookup_item_name(cdata, id);
}

// Translates a def_id from an external crate to a def_id for the current
// compilation environment. We use this when trying to load types from
// external crates - if those types further refer to types in other crates
// then we must translate the crate number from that encoded in the external
// crate to the correct local crate number.
fn translate_def_id(cstore: cstore::cstore, searched_crate: ast::crate_num,
                    def_id: ast::def_id) -> ast::def_id {

    let ext_cnum = def_id.crate;
    let node_id = def_id.node;

    assert (searched_crate != ast::local_crate);
    assert (ext_cnum != ast::local_crate);

    let cmeta = cstore::get_crate_data(cstore, searched_crate);

    let local_cnum =
        alt cmeta.cnum_map.find(ext_cnum) {
          option::some(n) { n }
          option::none. { fail "didn't find a crate in the cnum_map"; }
        };

    ret {crate: local_cnum, node: node_id};
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
