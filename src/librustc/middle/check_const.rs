// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::iterator::IteratorUtil;
use core::prelude::*;

use driver::session::Session;
use middle::resolve;
use middle::ty;
use middle::typeck;
use util::ppaux;

use syntax::ast::*;
use syntax::codemap;
use syntax::{visit, ast_util, ast_map};

pub fn check_crate(sess: Session,
                   crate: @crate,
                   ast_map: ast_map::map,
                   def_map: resolve::DefMap,
                   method_map: typeck::method_map,
                   tcx: ty::ctxt) {
    visit::visit_crate(crate, (false, visit::mk_vt(@visit::Visitor {
        visit_item: |a,b| check_item(sess, ast_map, def_map, a, b),
        visit_pat: check_pat,
        visit_expr: |a,b|
            check_expr(sess, def_map, method_map, tcx, a, b),
        .. *visit::default_visitor()
    })));
    sess.abort_if_errors();
}

pub fn check_item(sess: Session,
                  ast_map: ast_map::map,
                  def_map: resolve::DefMap,
                  it: @item,
                  (_is_const, v): (bool,
                                   visit::vt<bool>)) {
    match it.node {
      item_const(_, ex) => {
        (v.visit_expr)(ex, (true, v));
        check_item_recursion(sess, ast_map, def_map, it);
      }
      item_enum(ref enum_definition, _) => {
        for (*enum_definition).variants.each |var| {
            for var.node.disr_expr.iter().advance |ex| {
                (v.visit_expr)(*ex, (true, v));
            }
        }
      }
      _ => visit::visit_item(it, (false, v))
    }
}

pub fn check_pat(p: @pat, (_is_const, v): (bool, visit::vt<bool>)) {
    fn is_str(e: @expr) -> bool {
        match e.node {
            expr_vstore(
                @expr { node: expr_lit(@codemap::spanned {
                    node: lit_str(_),
                    _}),
                       _ },
                expr_vstore_uniq
            ) => true,
            _ => false
        }
    }
    match p.node {
      // Let through plain ~-string literals here
      pat_lit(a) => if !is_str(a) { (v.visit_expr)(a, (true, v)); },
      pat_range(a, b) => {
        if !is_str(a) { (v.visit_expr)(a, (true, v)); }
        if !is_str(b) { (v.visit_expr)(b, (true, v)); }
      }
      _ => visit::visit_pat(p, (false, v))
    }
}

pub fn check_expr(sess: Session,
                  def_map: resolve::DefMap,
                  method_map: typeck::method_map,
                  tcx: ty::ctxt,
                  e: @expr,
                  (is_const, v): (bool,
                                  visit::vt<bool>)) {
    if is_const {
        match e.node {
          expr_unary(_, deref, _) => { }
          expr_unary(_, box(_), _) | expr_unary(_, uniq(_), _) => {
            sess.span_err(e.span,
                          "disallowed operator in constant expression");
            return;
          }
          expr_lit(@codemap::spanned {node: lit_str(_), _}) => { }
          expr_binary(*) | expr_unary(*) => {
            if method_map.contains_key(&e.id) {
                sess.span_err(e.span, "user-defined operators are not \
                                       allowed in constant expressions");
            }
          }
          expr_lit(_) => (),
          expr_cast(_, _) => {
            let ety = ty::expr_ty(tcx, e);
            if !ty::type_is_numeric(ety) && !ty::type_is_unsafe_ptr(ety) {
                sess.span_err(e.span, ~"can not cast to `" +
                              ppaux::ty_to_str(tcx, ety) +
                              "` in a constant expression");
            }
          }
          expr_path(pth) => {
            // NB: In the future you might wish to relax this slightly
            // to handle on-demand instantiation of functions via
            // foo::<bar> in a const. Currently that is only done on
            // a path in trans::callee that only works in block contexts.
            if pth.types.len() != 0 {
                sess.span_err(
                    e.span, "paths in constants may only refer to \
                             items without type parameters");
            }
            match def_map.find(&e.id) {
              Some(&def_const(_)) |
              Some(&def_fn(_, _)) |
              Some(&def_variant(_, _)) |
              Some(&def_struct(_)) => { }

              Some(&def) => {
                debug!("(checking const) found bad def: %?", def);
                sess.span_err(
                    e.span,
                    "paths in constants may only refer to \
                     constants or functions");
              }
              None => {
                sess.span_bug(e.span, "unbound path in const?!");
              }
            }
          }
          expr_call(callee, _, NoSugar) => {
            match def_map.find(&callee.id) {
                Some(&def_struct(*)) => {}    // OK.
                Some(&def_variant(*)) => {}    // OK.
                _ => {
                    sess.span_err(
                        e.span,
                        "function calls in constants are limited to \
                         struct and enum constructors");
                }
            }
          }
          expr_paren(e) => { check_expr(sess, def_map, method_map,
                                         tcx, e, (is_const, v)); }
          expr_vstore(_, expr_vstore_slice) |
          expr_vec(_, m_imm) |
          expr_addr_of(m_imm, _) |
          expr_field(*) |
          expr_index(*) |
          expr_tup(*) |
          expr_struct(_, _, None) => { }
          expr_addr_of(*) => {
                sess.span_err(
                    e.span,
                    "borrowed pointers in constants may only refer to \
                     immutable values");
          }
          _ => {
            sess.span_err(e.span,
                          "constant contains unimplemented expression type");
            return;
          }
        }
    }
    match e.node {
      expr_lit(@codemap::spanned {node: lit_int(v, t), _}) => {
        if t != ty_char {
            if (v as u64) > ast_util::int_ty_max(
                if t == ty_i { sess.targ_cfg.int_type } else { t }) {
                sess.span_err(e.span, "literal out of range for its type");
            }
        }
      }
      expr_lit(@codemap::spanned {node: lit_uint(v, t), _}) => {
        if v > ast_util::uint_ty_max(
            if t == ty_u { sess.targ_cfg.uint_type } else { t }) {
            sess.span_err(e.span, "literal out of range for its type");
        }
      }
      _ => ()
    }
    visit::visit_expr(e, (is_const, v));
}

// Make sure a const item doesn't recursively refer to itself
// FIXME: Should use the dependency graph when it's available (#1356)
pub fn check_item_recursion(sess: Session,
                            ast_map: ast_map::map,
                            def_map: resolve::DefMap,
                            it: @item) {
    struct env {
        root_it: @item,
        sess: Session,
        ast_map: ast_map::map,
        def_map: resolve::DefMap,
        idstack: @mut ~[node_id]
    }

    let env = env {
        root_it: it,
        sess: sess,
        ast_map: ast_map,
        def_map: def_map,
        idstack: @mut ~[]
    };

    let visitor = visit::mk_vt(@visit::Visitor {
        visit_item: visit_item,
        visit_expr: visit_expr,
        .. *visit::default_visitor()
    });
    (visitor.visit_item)(it, (env, visitor));

    fn visit_item(it: @item, (env, v): (env, visit::vt<env>)) {
        if env.idstack.contains(&(it.id)) {
            env.sess.span_fatal(env.root_it.span, "recursive constant");
        }
        env.idstack.push(it.id);
        visit::visit_item(it, (env, v));
        env.idstack.pop();
    }

    fn visit_expr(e: @expr, (env, v): (env, visit::vt<env>)) {
        match e.node {
          expr_path(*) => {
            match env.def_map.find(&e.id) {
              Some(&def_const(def_id)) => {
                if ast_util::is_local(def_id) {
                  match env.ast_map.get_copy(&def_id.node) {
                    ast_map::node_item(it, _) => {
                      (v.visit_item)(it, (env, v));
                    }
                    _ => fail!("const not bound to an item")
                  }
                }
              }
              _ => ()
            }
          }
          _ => ()
        }
        visit::visit_expr(e, (env, v));
    }
}
