import driver::session;
import syntax::ast;
import middle::ty;
import std::io;

fn get_symbol(session::session sess, ast::def_id def) -> str {
    auto cnum = def._0;
    auto node_id = def._1;
    auto cstore = sess.get_cstore();
    auto cdata = cstore::get_crate_data(cstore, cnum).data;
    ret decoder::get_symbol(cdata, node_id);
}

fn get_tag_variants(ty::ctxt tcx, ast::def_id def) -> ty::variant_info[] {
    decoder::get_tag_variants(tcx, def)
}

fn get_type(ty::ctxt tcx, ast::def_id def) -> ty::ty_param_count_and_ty {
    decoder::get_type(tcx, def)
}

fn get_type_param_count(ty::ctxt tcx, &ast::def_id def) -> uint {
    auto cnum = def._0;
    auto node_id = def._1;
    auto cstore = tcx.sess.get_cstore();
    auto cdata = cstore::get_crate_data(cstore, cnum).data;
    ret decoder::get_type_param_count(cdata, node_id);
}

fn lookup_defs(session::session sess, ast::crate_num cnum,
               vec[ast::ident] path) -> vec[ast::def] {
    auto cstore = sess.get_cstore();
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
