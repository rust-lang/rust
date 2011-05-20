import std::vec;
import std::option;
import std::option::some;
import std::option::none;

import front::ast;
import front::ast::ident;
import front::ast::def_id;
import front::ast::ann;
import front::ast::item;
import front::ast::_fn;
import front::ast::_mod;
import front::ast::crate;
import front::ast::_obj;
import front::ast::ty_param;
import front::ast::item_fn;
import front::ast::item_obj;
import front::ast::item_ty;
import front::ast::item_tag;
import front::ast::item_const;
import front::ast::item_mod;
import front::ast::item_native_mod;
import front::ast::expr;
import front::ast::elt;
import front::ast::field;
import front::ast::decl;
import front::ast::decl_local;
import front::ast::decl_item;
import front::ast::initializer;
import front::ast::local;
import front::ast::arm;
import front::ast::expr_call;
import front::ast::expr_vec;
import front::ast::expr_tup;
import front::ast::expr_path;
import front::ast::expr_field;
import front::ast::expr_index;
import front::ast::expr_log;
import front::ast::expr_block;
import front::ast::expr_rec;
import front::ast::expr_if;
import front::ast::expr_binary;
import front::ast::expr_unary;
import front::ast::expr_assign;
import front::ast::expr_assign_op;
import front::ast::expr_while;
import front::ast::expr_do_while;
import front::ast::expr_alt;
import front::ast::expr_lit;
import front::ast::expr_ret;
import front::ast::expr_self_method;
import front::ast::expr_bind;
import front::ast::expr_spawn;
import front::ast::expr_ext;
import front::ast::expr_fail;
import front::ast::expr_break;
import front::ast::expr_cont;
import front::ast::expr_send;
import front::ast::expr_recv;
import front::ast::expr_put;
import front::ast::expr_port;
import front::ast::expr_chan;
import front::ast::expr_be;
import front::ast::expr_check;
import front::ast::expr_assert;
import front::ast::expr_cast;
import front::ast::expr_for;
import front::ast::expr_for_each;
import front::ast::stmt;
import front::ast::stmt_decl;
import front::ast::stmt_expr;
import front::ast::block;
import front::ast::block_;
import front::ast::method;

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
import aux::num_locals;
import aux::get_fn_info;
import aux::crate_ctxt;
import aux::add_node;
import middle::tstate::ann::empty_ann;

fn collect_ids_expr(&@expr e, @vec[uint] res) -> () {
    vec::push(*res, (expr_ann(e)).id);
}
fn collect_ids_block(&block b, @vec[uint] res) -> () {
    vec::push(*res, b.node.a.id);
}

fn collect_ids_stmt(&@stmt s, @vec[uint] res) -> () {
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

fn collect_ids_decl(&@decl d, @vec[uint] res) -> () {
    alt (d.node) {
        case (decl_local(?l)) {
            vec::push(*res, l.ann.id);
        }
        case (_) {}
    }
}

fn node_ids_in_fn(&_fn f, &ident i, &def_id d, &ann a, @vec[uint] res) -> () {
    auto collect_ids = walk::default_visitor();
    collect_ids = rec(visit_expr_pre  = bind collect_ids_expr(_,res),
                      visit_block_pre = bind collect_ids_block(_,res),
                      visit_stmt_pre  = bind collect_ids_stmt(_,res),
                      visit_decl_pre  = bind collect_ids_decl(_,res)
                      with collect_ids);
    walk::walk_fn(collect_ids, f, i, d, a);
}

fn init_vecs(&crate_ctxt ccx, @vec[uint] node_ids, uint len) -> () {
    for (uint i in *node_ids) {
        log(uistr(i) + " |-> " + uistr(len));
        add_node(ccx, i, empty_ann(len));
    }
}

fn visit_fn(&crate_ctxt ccx, uint num_locals, &_fn f, &ident i,
            &def_id d, &ann a) -> () {
    let vec[uint] node_ids_ = [];
    let @vec[uint] node_ids = @node_ids_;
    node_ids_in_fn(f, i, d, a, node_ids);
    init_vecs(ccx, node_ids, num_locals);
}

fn annotate_in_fn(&crate_ctxt ccx, &_fn f, &ident i, &def_id f_id, &ann a)
    -> () {
    auto f_info = get_fn_info(ccx, f_id);
    visit_fn(ccx, num_locals(f_info), f, i, f_id, a);
}

fn annotate_crate(&crate_ctxt ccx, &crate crate) -> () {
    auto do_ann = walk::default_visitor();
    do_ann = rec(visit_fn_pre = bind annotate_in_fn(ccx,_,_,_,_)
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
