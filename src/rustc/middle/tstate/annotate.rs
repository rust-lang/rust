
import syntax::ast::*;
import syntax::visit;
import syntax::codemap::span;
import syntax::print::pprust::stmt_to_str;
import aux::{num_constraints, get_fn_info, crate_ctxt, add_node};
import ann::empty_ann;
import pat_util::pat_binding_ids;

fn collect_ids_expr(e: @expr, rs: @mut ~[node_id]) { vec::push(*rs, e.id); }

fn collect_ids_block(b: blk, rs: @mut ~[node_id]) {
    vec::push(*rs, b.node.id);
}

fn collect_ids_stmt(s: @stmt, rs: @mut ~[node_id]) {
    alt s.node {
      stmt_decl(_, id) | stmt_expr(_, id) {
        #debug["node_id %s", int::str(id)];
        #debug["%s", stmt_to_str(*s)];
        vec::push(*rs, id);
      }
    }
}

fn collect_ids_local(tcx: ty::ctxt, l: @local, rs: @mut ~[node_id]) {
    vec::push_all(*rs, pat_binding_ids(tcx.def_map, l.node.pat));
}

fn node_ids_in_fn(tcx: ty::ctxt, body: blk, rs: @mut ~[node_id]) {
    let collect_ids =
        visit::mk_simple_visitor(@{visit_expr: |a| collect_ids_expr(a, rs),
                                   visit_block: |a| collect_ids_block(a, rs),
                                   visit_stmt: |a| collect_ids_stmt(a, rs),
                                   visit_local: |a|
                                       collect_ids_local(tcx, a, rs)
                                   with *visit::default_simple_visitor()});
    collect_ids.visit_block(body, (), collect_ids);
}

fn init_vecs(ccx: crate_ctxt, node_ids: ~[node_id], len: uint) {
    for node_ids.each |i| {
        log(debug, int::str(i) + " |-> " + uint::str(len));
        add_node(ccx, i, empty_ann(len));
    }
}

fn visit_fn(ccx: crate_ctxt, num_constraints: uint, body: blk) {
    let node_ids: @mut ~[node_id] = @mut ~[];
    node_ids_in_fn(ccx.tcx, body, node_ids);
    let node_id_vec = *node_ids;
    init_vecs(ccx, node_id_vec, num_constraints);
}

fn annotate_in_fn(ccx: crate_ctxt, _fk: visit::fn_kind, _decl: fn_decl,
                  body: blk, _sp: span, id: node_id) {
    let f_info = get_fn_info(ccx, id);
    visit_fn(ccx, num_constraints(f_info), body);
}

fn annotate_crate(ccx: crate_ctxt, crate: crate) {
    let do_ann =
        visit::mk_simple_visitor(
            @{visit_fn: |a,b,c,d,e| annotate_in_fn(ccx, a, b, c, d, e)
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
// End:
//
