import trans::*;
import trans_common::*;
import option::{some, none};
import syntax::ast;

fn trans_impl(cx: @local_ctxt, name: ast::ident, methods: [@ast::method],
              id: ast::node_id, tps: [ast::ty_param]) {
    let sub_cx = extend_path(cx, name);
    for m in methods {
        alt cx.ccx.item_ids.find(m.id) {
          some(llfn) {
            trans_fn(extend_path(sub_cx, m.ident), m.span, m.decl, m.body,
                     llfn, impl_self(ty::node_id_to_monotype(cx.ccx.tcx, id)),
                     tps + m.tps, m.id);
          }
        }
    }
}
