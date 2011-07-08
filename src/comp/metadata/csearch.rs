// Searching for information from the cstore

import syntax::ast;
import middle::ty;
import std::io;
import std::option;
import driver::session;

export get_symbol;
export get_type_param_count;
export lookup_defs;
export get_tag_variants;
export get_type;

fn get_symbol(&cstore::cstore cstore, ast::def_id def) -> str {
    auto cnum = def._0;
    auto node_id = def._1;
    auto cdata = cstore::get_crate_data(cstore, cnum).data;
    ret decoder::get_symbol(cdata, node_id);
}

fn get_type_param_count(&cstore::cstore cstore, &ast::def_id def) -> uint {
    auto cnum = def._0;
    auto node_id = def._1;
    auto cdata = cstore::get_crate_data(cstore, cnum).data;
    ret decoder::get_type_param_count(cdata, node_id);
}

fn lookup_defs(&cstore::cstore cstore, ast::crate_num cnum,
               vec[ast::ident] path) -> vec[ast::def] {
    auto cdata = cstore::get_crate_data(cstore, cnum).data;
    ret decoder::lookup_defs(cdata, cnum, path);
}

fn get_tag_variants(ty::ctxt tcx, ast::def_id def) -> ty::variant_info[] {
    auto cstore = tcx.sess.get_cstore();
    auto cnum = def._0;
    auto cdata = cstore::get_crate_data(cstore, cnum).data;
    auto resolver = bind translate_def_id(tcx.sess, cnum, _);
    ret decoder::get_tag_variants(cdata, def, tcx, resolver)
}

fn get_type(ty::ctxt tcx, ast::def_id def) -> ty::ty_param_count_and_ty {
    auto cstore = tcx.sess.get_cstore();
    auto cnum = def._0;
    auto cdata = cstore::get_crate_data(cstore, cnum).data;
    auto resolver = bind translate_def_id(tcx.sess, cnum, _);
    decoder::get_type(cdata, def, tcx, resolver)
}

// Translates a def_id from an external crate to a def_id for the current
// compilation environment. We use this when trying to load types from
// external crates - if those types further refer to types in other crates
// then we must translate the crate number from that encoded in the external
// crate to the correct local crate number.
fn translate_def_id(&session::session sess,
                    ast::crate_num searched_crate,
                    &ast::def_id def_id) -> ast::def_id {

    auto ext_cnum = def_id._0;
    auto node_id = def_id._1;

    assert searched_crate != ast::local_crate;
    assert ext_cnum != ast::local_crate;

    auto cstore = sess.get_cstore();
    auto cmeta = cstore::get_crate_data(cstore, searched_crate);

    auto local_cnum = alt (cmeta.cnum_map.find(ext_cnum)) {
        case (option::some(?n)) { n }
        case (option::none) {
            sess.bug("didn't find a crate in the cnum_map")
        }
    };

    ret tup(local_cnum, node_id);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
