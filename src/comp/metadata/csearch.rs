import driver::session;
import syntax::ast;
import middle::ty;
import std::io;

fn get_symbol(session::session sess, ast::def_id def) -> str {
    decoder::get_symbol(sess, def)
}

fn get_tag_variants(ty::ctxt ctx, ast::def_id def) -> ty::variant_info[] {
    decoder::get_tag_variants(ctx, def)
}

fn get_type(ty::ctxt tcx, ast::def_id def) -> ty::ty_param_count_and_ty {
    decoder::get_type(tcx, def)
}

fn get_type_param_count(ty::ctxt tcx, &ast::def_id def) -> uint {
    decoder::get_type_param_count(tcx, def)
}

fn lookup_defs(session::session sess, ast::crate_num cnum,
               vec[ast::ident] path) -> vec[ast::def] {
    decoder::lookup_defs(sess, cnum, path)
}

fn get_crate_attributes(&vec[u8] data) -> ast::attribute[] {
    decoder::get_crate_attributes(data)
}

fn list_crate_metadata(vec[u8] data, io::writer out) {
    decoder::list_crate_metadata(data, out)
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
