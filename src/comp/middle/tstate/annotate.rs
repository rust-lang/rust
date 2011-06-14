import std::vec;
import std::option;
import std::option::some;
import std::option::none;

import front::ast::*;
import middle::ty::expr_ann;

import util::common::uistr;
import util::common::span;
import util::common::new_str_hash;
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

fn collect_ids_expr(&@expr e, @mutable vec[uint] res) -> () {
    vec::push(*res, (expr_ann(e)).id);
}
fn collect_ids_block(&block b, @mutable vec[uint] res) -> () {
    vec::push(*res, b.node.a.id);
}

fn collect_ids_stmt(&@stmt s, @mutable vec[uint] res) -> () {
    alt (s.node) {
        case (stmt_decl(_,?a)) {
            log("node_id " + uistr(a.id));
            log_stmt(*s);
  
            vec::push(*res, a.id);
        }
        case (stmt_expr(_,?a)) {
            log("node_id " + uistr(a.id));
            log_stmt(*s);
    
            vec::push(*res, a.id);
        }
        case (_) {}
    }
}

fn collect_ids_local(&@local l, @mutable vec[uint] res) -> () {
    vec::push(*res, l.node.ann.id);
}

fn node_ids_in_fn(&_fn f, &span sp, &ident i, &def_id d, &ann a,
                  @mutable vec[uint] res) -> () {
    auto collect_ids = walk::default_visitor();
    collect_ids = rec(visit_expr_pre  = bind collect_ids_expr(_,res),
                      visit_block_pre = bind collect_ids_block(_,res),
                      visit_stmt_pre  = bind collect_ids_stmt(_,res),
                      visit_local_pre  = bind collect_ids_local(_,res)
                      with collect_ids);
    walk::walk_fn(collect_ids, f, sp, i, d, a);
}

fn init_vecs(&crate_ctxt ccx, &vec[uint] node_ids, uint len) -> () {
    for (uint i in node_ids) {
        log(uistr(i) + " |-> " + uistr(len));
        add_node(ccx, i, empty_ann(len));
    }
}

fn visit_fn(&crate_ctxt ccx, uint num_constraints, &_fn f,
            &span sp, &ident i, &def_id d, &ann a) -> () {
    let @mutable vec[uint] node_ids = @mutable [];
    node_ids_in_fn(f, sp, i, d, a, node_ids);
    auto node_id_vec = *node_ids;
    init_vecs(ccx, node_id_vec, num_constraints);
}

fn annotate_in_fn(&crate_ctxt ccx, &_fn f, &span sp, &ident i,
                  &def_id f_id, &ann a)
    -> () {
    auto f_info = get_fn_info(ccx, f_id);
    visit_fn(ccx, num_constraints(f_info), f, sp, i, f_id, a);
}

fn annotate_crate(&crate_ctxt ccx, &crate crate) -> () {
    auto do_ann = walk::default_visitor();
    do_ann = rec(visit_fn_pre = bind annotate_in_fn(ccx,_,_,_,_,_)
                 with do_ann);
    walk::walk_crate(do_ann, crate);
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
