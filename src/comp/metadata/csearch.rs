// Searching for information from the cstore

import syntax::ast;
import middle::ty;
import std::option;
import driver::session;

export get_symbol;
export get_type_param_count;
export lookup_defs;
export get_tag_variants;
export get_type;

fn get_symbol(cstore: &cstore::cstore, def: ast::def_id) -> istr {
    let cdata = cstore::get_crate_data(cstore, def.crate).data;
    ret decoder::get_symbol(cdata, def.node);
}

fn get_type_param_count(cstore: &cstore::cstore, def: &ast::def_id) -> uint {
    let cdata = cstore::get_crate_data(cstore, def.crate).data;
    ret decoder::get_type_param_count(cdata, def.node);
}

fn lookup_defs(cstore: &cstore::cstore, cnum: ast::crate_num,
               path: &[ast::ident]) -> [ast::def] {
    let cdata = cstore::get_crate_data(cstore, cnum).data;
    ret decoder::lookup_defs(cdata, cnum, path);
}

fn get_tag_variants(tcx: ty::ctxt, def: ast::def_id) -> [ty::variant_info] {
    let cstore = tcx.sess.get_cstore();
    let cnum = def.crate;
    let cdata = cstore::get_crate_data(cstore, cnum).data;
    let resolver = bind translate_def_id(tcx.sess, cnum, _);
    ret decoder::get_tag_variants(cdata, def, tcx, resolver)
}

fn get_type(tcx: ty::ctxt, def: ast::def_id) -> ty::ty_param_kinds_and_ty {
    let cstore = tcx.sess.get_cstore();
    let cnum = def.crate;
    let cdata = cstore::get_crate_data(cstore, cnum).data;
    let resolver = bind translate_def_id(tcx.sess, cnum, _);
    decoder::get_type(cdata, def, tcx, resolver)
}

// Translates a def_id from an external crate to a def_id for the current
// compilation environment. We use this when trying to load types from
// external crates - if those types further refer to types in other crates
// then we must translate the crate number from that encoded in the external
// crate to the correct local crate number.
fn translate_def_id(sess: &session::session, searched_crate: ast::crate_num,
                    def_id: &ast::def_id) -> ast::def_id {

    let ext_cnum = def_id.crate;
    let node_id = def_id.node;

    assert (searched_crate != ast::local_crate);
    assert (ext_cnum != ast::local_crate);

    let cstore = sess.get_cstore();
    let cmeta = cstore::get_crate_data(cstore, searched_crate);

    let local_cnum =
        alt cmeta.cnum_map.find(ext_cnum) {
          option::some(n) { n }
          option::none. { sess.bug("didn't find a crate in the cnum_map") }
        };

    ret {crate: local_cnum, node: node_id};
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
