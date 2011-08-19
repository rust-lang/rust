
import std::option;
import std::option::some;
import std::option::none;
import std::int;
import std::uint;
import syntax::ast::*;
import syntax::visit;
import syntax::codemap::span;
import std::map::new_str_hash;
import util::common::log_expr_err;
import util::common::log_block_err;
import util::common::log_item_err;
import util::common::log_stmt_err;
import util::common::log_expr;
import util::common::log_block;
import util::common::log_stmt;
import aux::fn_info;
import aux::fn_info_map;
import aux::num_constraints;
import aux::get_fn_info;
import aux::crate_ctxt;
import aux::add_node;
import middle::tstate::ann::empty_ann;

fn collect_ids_expr(e: &@expr, rs: @mutable [node_id]) { *rs += [e.id]; }

fn collect_ids_block(b: &blk, rs: @mutable [node_id]) { *rs += [b.node.id]; }

fn collect_ids_stmt(s: &@stmt, rs: @mutable [node_id]) {
    alt s.node {
      stmt_decl(_, id) {
        log "node_id " + int::str(id);
        log_stmt(*s);;
        *rs += [id];
      }
      stmt_expr(_, id) {
        log "node_id " + int::str(id);
        log_stmt(*s);;
        *rs += [id];
      }
      _ { }
    }
}

fn collect_ids_local(l: &@local, rs: @mutable [node_id]) {
    *rs += pat_binding_ids(l.node.pat);
}

fn node_ids_in_fn(f: &_fn, tps: &[ty_param], sp: &span, i: &fn_ident,
                  id: node_id, rs: @mutable [node_id]) {
    let collect_ids =
        visit::mk_simple_visitor(@{visit_expr: bind collect_ids_expr(_, rs),
                                   visit_block: bind collect_ids_block(_, rs),
                                   visit_stmt: bind collect_ids_stmt(_, rs),
                                   visit_local: bind collect_ids_local(_, rs)
                                      with *visit::default_simple_visitor()});
    visit::visit_fn(f, tps, sp, i, id, (), collect_ids);
}

fn init_vecs(ccx: &crate_ctxt, node_ids: &[node_id], len: uint) {
    for i: node_id in node_ids {
        log int::str(i) + " |-> " + uint::str(len);
        add_node(ccx, i, empty_ann(len));
    }
}

fn visit_fn(ccx: &crate_ctxt, num_constraints: uint, f: &_fn,
            tps: &[ty_param], sp: &span, i: &fn_ident, id: node_id) {
    let node_ids: @mutable [node_id] = @mutable [];
    node_ids_in_fn(f, tps, sp, i, id, node_ids);
    let node_id_vec = *node_ids;
    init_vecs(ccx, node_id_vec, num_constraints);
}

fn annotate_in_fn(ccx: &crate_ctxt, f: &_fn, tps: &[ty_param], sp: &span,
                  i: &fn_ident, id: node_id) {
    let f_info = get_fn_info(ccx, id);
    visit_fn(ccx, num_constraints(f_info), f, tps, sp, i, id);
}

fn annotate_crate(ccx: &crate_ctxt, crate: &crate) {
    let do_ann =
        visit::mk_simple_visitor(@{visit_fn:
                                       bind annotate_in_fn(ccx, _, _, _, _, _)
                                      with *visit::default_simple_visitor()});
    visit::visit_crate(crate, (), do_ann);
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
