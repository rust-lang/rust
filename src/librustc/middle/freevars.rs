// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A pass that annotates for each loops and functions with the free
// variables that they contain.


use middle::resolve;
use middle::ty;

use std::hashmap::HashMap;
use syntax::codemap::span;
use syntax::{ast, ast_util, visit};

// A vector of defs representing the free variables referred to in a function.
// (The def_upvar will already have been stripped).
#[deriving(Encodable, Decodable)]
pub struct freevar_entry {
    def: ast::def, //< The variable being accessed free.
    span: span     //< First span where it is accessed (there can be multiple)
}
pub type freevar_info = @~[@freevar_entry];
pub type freevar_map = @mut HashMap<ast::node_id, freevar_info>;

// Searches through part of the AST for all references to locals or
// upvars in this frame and returns the list of definition IDs thus found.
// Since we want to be able to collect upvars in some arbitrary piece
// of the AST, we take a walker function that we invoke with a visitor
// in order to start the search.
fn collect_freevars(def_map: resolve::DefMap, blk: &ast::blk)
    -> freevar_info {
    let seen = @mut HashMap::new();
    let refs = @mut ~[];

    fn ignore_item(_i: @ast::item, (_depth, _v): (int, visit::vt<int>)) { }

    let walk_expr: @fn(expr: @ast::expr, (int, visit::vt<int>)) =
        |expr, (depth, v)| {
            match expr.node {
              ast::expr_fn_block(*) => visit::visit_expr(expr, (depth + 1, v)),
              ast::expr_path(*) | ast::expr_self => {
                  let mut i = 0;
                  match def_map.find(&expr.id) {
                    None => fail!("path not found"),
                    Some(&df) => {
                      let mut def = df;
                      while i < depth {
                        match def {
                          ast::def_upvar(_, inner, _, _) => { def = *inner; }
                          _ => break
                        }
                        i += 1;
                      }
                      if i == depth { // Made it to end of loop
                        let dnum = ast_util::def_id_of_def(def).node;
                        if !seen.contains_key(&dnum) {
                            refs.push(@freevar_entry {
                                def: def,
                                span: expr.span,
                            });
                            seen.insert(dnum, ());
                        }
                      }
                    }
                  }
              }
              _ => visit::visit_expr(expr, (depth, v))
            }
        };

    let v = visit::mk_vt(@visit::Visitor {visit_item: ignore_item,
                                          visit_expr: walk_expr,
                                          .. *visit::default_visitor()});
    (v.visit_block)(blk, (1, v));
    return @/*bad*/copy *refs;
}

// Build a map from every function and for-each body to a set of the
// freevars contained in it. The implementation is not particularly
// efficient as it fully recomputes the free variables at every
// node of interest rather than building up the free variables in
// one pass. This could be improved upon if it turns out to matter.
pub fn annotate_freevars(def_map: resolve::DefMap, crate: &ast::crate) ->
   freevar_map {
    let freevars = @mut HashMap::new();

    let walk_fn: @fn(&visit::fn_kind,
                     &ast::fn_decl,
                     &ast::blk,
                     span,
                     ast::node_id) = |_, _, blk, _, nid| {
        let vars = collect_freevars(def_map, blk);
        freevars.insert(nid, vars);
    };

    let visitor =
        visit::mk_simple_visitor(@visit::SimpleVisitor {
            visit_fn: walk_fn,
            .. *visit::default_simple_visitor()});
    visit::visit_crate(crate, ((), visitor));

    return freevars;
}

pub fn get_freevars(tcx: ty::ctxt, fid: ast::node_id) -> freevar_info {
    match tcx.freevars.find(&fid) {
      None => fail!("get_freevars: %d has no freevars", fid),
      Some(&d) => return d
    }
}

pub fn has_freevars(tcx: ty::ctxt, fid: ast::node_id) -> bool {
    !get_freevars(tcx, fid).is_empty()
}
