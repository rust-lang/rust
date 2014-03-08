// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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

#[allow(non_camel_case_types)];

use middle::resolve;
use middle::ty;
use util::nodemap::{NodeMap, NodeSet};

use std::vec_ng::Vec;
use syntax::codemap::Span;
use syntax::{ast, ast_util};
use syntax::visit;
use syntax::visit::Visitor;

// A vector of defs representing the free variables referred to in a function.
// (The def_upvar will already have been stripped).
#[deriving(Encodable, Decodable)]
pub struct freevar_entry {
    def: ast::Def, //< The variable being accessed free.
    span: Span     //< First span where it is accessed (there can be multiple)
}
pub type freevar_info = @Vec<@freevar_entry> ;
pub type freevar_map = NodeMap<freevar_info>;

struct CollectFreevarsVisitor {
    seen: NodeSet,
    refs: Vec<@freevar_entry> ,
    def_map: resolve::DefMap,
}

impl Visitor<int> for CollectFreevarsVisitor {

    fn visit_item(&mut self, _: &ast::Item, _: int) {
        // ignore_item
    }

    fn visit_expr(&mut self, expr: &ast::Expr, depth: int) {
        match expr.node {
            ast::ExprFnBlock(..) | ast::ExprProc(..) => {
                visit::walk_expr(self, expr, depth + 1)
            }
            ast::ExprPath(..) => {
                let mut i = 0;
                let def_map = self.def_map.borrow();
                match def_map.get().find(&expr.id) {
                    None => fail!("path not found"),
                    Some(&df) => {
                        let mut def = df;
                        while i < depth {
                            match def {
                                ast::DefUpvar(_, inner, _, _) => { def = *inner; }
                                _ => break
                            }
                            i += 1;
                        }
                        if i == depth { // Made it to end of loop
                            let dnum = ast_util::def_id_of_def(def).node;
                            if !self.seen.contains(&dnum) {
                                self.refs.push(@freevar_entry {
                                    def: def,
                                    span: expr.span,
                                });
                                self.seen.insert(dnum);
                            }
                        }
                    }
                }
            }
            _ => visit::walk_expr(self, expr, depth)
        }
    }


}

// Searches through part of the AST for all references to locals or
// upvars in this frame and returns the list of definition IDs thus found.
// Since we want to be able to collect upvars in some arbitrary piece
// of the AST, we take a walker function that we invoke with a visitor
// in order to start the search.
fn collect_freevars(def_map: resolve::DefMap, blk: &ast::Block) -> freevar_info {
    let seen = NodeSet::new();
    let refs = Vec::new();

    let mut v = CollectFreevarsVisitor {
        seen: seen,
        refs: refs,
        def_map: def_map,
    };

    v.visit_block(blk, 1);
    let CollectFreevarsVisitor {
        refs,
        ..
    } = v;
    return @refs;
}

struct AnnotateFreevarsVisitor {
    def_map: resolve::DefMap,
    freevars: freevar_map,
}

impl Visitor<()> for AnnotateFreevarsVisitor {
    fn visit_fn(&mut self, fk: &visit::FnKind, fd: &ast::FnDecl,
                blk: &ast::Block, s: Span, nid: ast::NodeId, _: ()) {
        let vars = collect_freevars(self.def_map, blk);
        self.freevars.insert(nid, vars);
        visit::walk_fn(self, fk, fd, blk, s, nid, ());
    }
}

// Build a map from every function and for-each body to a set of the
// freevars contained in it. The implementation is not particularly
// efficient as it fully recomputes the free variables at every
// node of interest rather than building up the free variables in
// one pass. This could be improved upon if it turns out to matter.
pub fn annotate_freevars(def_map: resolve::DefMap, krate: &ast::Crate) ->
   freevar_map {
    let mut visitor = AnnotateFreevarsVisitor {
        def_map: def_map,
        freevars: NodeMap::new(),
    };
    visit::walk_crate(&mut visitor, krate, ());

    let AnnotateFreevarsVisitor {
        freevars,
        ..
    } = visitor;
    freevars
}

pub fn get_freevars(tcx: ty::ctxt, fid: ast::NodeId) -> freevar_info {
    let freevars = tcx.freevars.borrow();
    match freevars.get().find(&fid) {
        None => fail!("get_freevars: {} has no freevars", fid),
        Some(&d) => return d
    }
}

pub fn has_freevars(tcx: ty::ctxt, fid: ast::NodeId) -> bool {
    !get_freevars(tcx, fid).is_empty()
}
