
import codemap::emit_error;
import driver::session;
import syntax::ast::crate;
import syntax::ast::expr_;
import syntax::ast::expr_mac;
import syntax::ast::mac_invoc;
import syntax::fold::*;

import std::option::none;
import std::option::some;

import std::map::hashmap;
import std::ivec;

fn expand_expr(exts: &hashmap[str, base::syntax_extension],
               sess: &session::session, e: &expr_, fld: ast_fold,
               orig: &fn(&ast::expr_, ast_fold) -> expr_ ) -> expr_ {
    ret alt e {
          expr_mac(mac) {
            alt mac.node {
              mac_invoc(pth, args, body) {
                assert (ivec::len(pth.node.idents) > 0u);
                let extname = pth.node.idents.(0);
                let ext_cx = base::mk_ctxt(sess);
                alt exts.find(extname) {
                  none. {
                    emit_error(some(pth.span),
                               "unknown syntax expander: '" + extname + "'",
                               sess.get_codemap());
                    fail
                  }
                  some(base::normal(ext)) {

                    //keep going, outside-in
                    fld.fold_expr(ext(ext_cx, pth.span, args, body)).node
                  }
                  some(base::macro_defining(ext)) {
                    let named_extension = ext(ext_cx, pth.span, args, body);
                    exts.insert(named_extension.ident, named_extension.ext);
                    ast::expr_rec(~[], none)
                  }
                }
              }
              _ {
                emit_error(some(mac.span), "naked syntactic bit",
                           sess.get_codemap());
                fail
              }
            }
          }
          _ { orig(e, fld) }
        };
}

fn expand_crate(sess: &session::session, c: &@crate) -> @crate {
    let exts = ext::base::syntax_expander_table();
    let afp = default_ast_fold();
    let f_pre =
        {fold_expr: bind expand_expr(exts, sess, _, _, afp.fold_expr)
            with *afp};
    let f = make_fold(f_pre);
    let res = @f.fold_crate(*c);
    dummy_out(f); //temporary: kill circular reference
    ret res;

}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
