// Searching for information from the cstore

import syntax::ast;
import middle::ty;
import std::io;

fn get_symbol(&cstore::cstore cstore, ast::def_id def) -> str {
    auto cnum = def._0;
    auto node_id = def._1;
    auto cdata = cstore::get_crate_data(cstore, cnum).data;
    ret decoder::get_symbol(cdata, node_id);
}

fn get_tag_variants(ty::ctxt tcx, ast::def_id def) -> ty::variant_info[] {
    auto cstore = tcx.sess.get_cstore();
    auto cnum = def._0;
    auto cdata = cstore::get_crate_data(cstore, cnum).data;
    ret decoder::get_tag_variants(cdata, def, tcx)
}

fn get_type(ty::ctxt tcx, ast::def_id def) -> ty::ty_param_count_and_ty {
    auto cstore = tcx.sess.get_cstore();
    auto cnum = def._0;
    auto cdata = cstore::get_crate_data(cstore, cnum).data;
    decoder::get_type(cdata, def, tcx)
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

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
