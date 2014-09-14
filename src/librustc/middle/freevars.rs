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

#![allow(non_camel_case_types)]

use middle::def;
use middle::mem_categorization::Typer;
use middle::resolve;
use middle::ty;
use util::nodemap::{NodeMap, NodeSet};

use syntax::ast;
use syntax::codemap::Span;
use syntax::visit::Visitor;
use syntax::visit;

#[deriving(Clone, Decodable, Encodable, Show)]
pub enum CaptureMode {
    /// Copy/move the value into the environment.
    CaptureByValue,

    /// Access by reference (used for stack closures).
    CaptureByRef
}

// A vector of defs representing the free variables referred to in a function.
// (The def_upvar will already have been stripped).
#[deriving(Encodable, Decodable)]
pub struct freevar_entry {
    pub def: def::Def, //< The variable being accessed free.
    pub span: Span     //< First span where it is accessed (there can be multiple)
}

pub type freevar_map = NodeMap<Vec<freevar_entry>>;

pub type CaptureModeMap = NodeMap<CaptureMode>;

struct CollectFreevarsVisitor<'a, 'b:'a> {
    node_id: ast::NodeId,
    seen: NodeSet,
    cx: &'a mut AnnotateFreevarsVisitor<'b>,
    depth: u32
}

impl<'a, 'b, 'v> Visitor<'v> for CollectFreevarsVisitor<'a, 'b> {
    fn visit_item(&mut self, _: &ast::Item) {
        // ignore_item
    }

    fn visit_expr(&mut self, expr: &ast::Expr) {
        match expr.node {
            ast::ExprProc(..) => {
                self.cx.capture_mode_map.insert(expr.id, CaptureByValue);
                self.depth += 1;
                visit::walk_expr(self, expr);
                self.depth -= 1;
            }
            ast::ExprFnBlock(_, _, _) => {
                // NOTE(stage0): After snapshot, change to:
                //
                //let capture_mode = match capture_clause {
                //    ast::CaptureByValue => CaptureByValue,
                //    ast::CaptureByRef => CaptureByRef,
                //};
                let capture_mode = CaptureByRef;
                self.cx.capture_mode_map.insert(expr.id, capture_mode);
                self.depth += 1;
                visit::walk_expr(self, expr);
                self.depth -= 1;
            }
            ast::ExprUnboxedFn(capture_clause, _, _, _) => {
                let capture_mode = match capture_clause {
                    ast::CaptureByValue => CaptureByValue,
                    ast::CaptureByRef => CaptureByRef,
                };
                self.cx.capture_mode_map.insert(expr.id, capture_mode);
                self.depth += 1;
                visit::walk_expr(self, expr);
                self.depth -= 1;
            }
            ast::ExprPath(..) => {
                let def = *self.cx.def_map.borrow().find(&expr.id)
                                                   .expect("path not found");
                let dnum = def.def_id().node;
                if self.seen.contains(&dnum) {
                    return;
                }
                let def = match def {
                    def::DefUpvar(_, _, depth, _, _) => {
                        if depth < self.depth {
                            return;
                        }
                        let mut def = def;
                        for _ in range(0, depth - self.depth) {
                            match def {
                                def::DefUpvar(_, inner, _, _, _) => { def = *inner; }
                                _ => unreachable!()
                            }
                        }
                        def
                    },
                    _ => return
                };
                self.cx.freevars.find_or_insert(self.node_id, vec![]).push(freevar_entry {
                    def: def,
                    span: expr.span,
                });
                self.seen.insert(dnum);
            }
            _ => visit::walk_expr(self, expr)
        }
    }
}

struct AnnotateFreevarsVisitor<'a> {
    def_map: &'a resolve::DefMap,
    freevars: freevar_map,
    capture_mode_map: CaptureModeMap,
}

impl<'a, 'v> Visitor<'v> for AnnotateFreevarsVisitor<'a> {
    fn visit_fn(&mut self, fk: visit::FnKind<'v>, fd: &'v ast::FnDecl,
                blk: &'v ast::Block, s: Span, nid: ast::NodeId) {
        CollectFreevarsVisitor {
            node_id: nid,
            seen: NodeSet::new(),
            cx: self,
            depth: 0
        }.visit_block(blk);
        visit::walk_fn(self, fk, fd, blk, s);
    }
}

// Build a map from every function and for-each body to a set of the
// freevars contained in it. The implementation is not particularly
// efficient as it fully recomputes the free variables at every
// node of interest rather than building up the free variables in
// one pass. This could be improved upon if it turns out to matter.
pub fn annotate_freevars(def_map: &resolve::DefMap, krate: &ast::Crate)
                         -> (freevar_map, CaptureModeMap) {
    let mut visitor = AnnotateFreevarsVisitor {
        def_map: def_map,
        freevars: NodeMap::new(),
        capture_mode_map: NodeMap::new(),
    };
    visit::walk_crate(&mut visitor, krate);
    (visitor.freevars, visitor.capture_mode_map)
}

pub fn with_freevars<T>(tcx: &ty::ctxt, fid: ast::NodeId, f: |&[freevar_entry]| -> T) -> T {
    match tcx.freevars.borrow().find(&fid) {
        None => fail!("with_freevars: {} has no freevars", fid),
        Some(d) => f(d.as_slice())
    }
}

pub fn get_capture_mode<'tcx, T:Typer<'tcx>>(tcx: &T, closure_expr_id: ast::NodeId)
                                             -> CaptureMode {
    tcx.capture_mode(closure_expr_id)
}
