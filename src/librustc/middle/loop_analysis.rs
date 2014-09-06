// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A loop nesting tree.

use middle::graph::{Graph, NodeIndex};
use middle::ty::ctxt;
use util::nodemap::NodeMap;

use syntax::ast::{Block, Decl, DeclItem, DeclLocal, Expr, ExprAddrOf};
use syntax::ast::{ExprAgain, ExprAssign, ExprAssignOp, ExprBinary, ExprBlock};
use syntax::ast::{ExprBox, ExprBreak, ExprCall, ExprCast, ExprField};
use syntax::ast::{ExprFnBlock, ExprForLoop, ExprIf, ExprIndex, ExprInlineAsm};
use syntax::ast::{ExprLit, ExprLoop, ExprMac, ExprMatch, ExprMethodCall};
use syntax::ast::{ExprParen, ExprPath, ExprProc, ExprRepeat, ExprRet};
use syntax::ast::{ExprStruct, ExprTup, ExprUnary, ExprUnboxedFn, ExprVec};
use syntax::ast::{ExprWhile, NodeId, Stmt, StmtDecl, StmtExpr, StmtMac};
use syntax::ast::{StmtSemi};

pub struct LoopAnalysis {
    /// A loop nesting tree.
    graph: Graph<LoopAnalysisNode,()>,
    /// The outermost loop node.
    entry: NodeIndex,
    /// A mapping from scope ID to index in the graph.
    scope_map: NodeMap<NodeIndex>,
}

#[allow(dead_code)]
pub struct LoopAnalysisNode {
    /// The ID of the expression that defines the loop.
    expr_id: NodeId,
    /// The ID of the scope that encompasses the loop (i.e. the loop body).
    scope_id: NodeId,
    /// The kind of loop that this is.
    kind: LoopKind,
    /// The loop exits immediately nested inside this loop. Note that this
    /// may not encompass *all* the exits of this loop, because of labeled
    /// break and continue as well as early return. To find all the exits of
    /// a loop, check for all of its descendants' exits as well.
    exits: Vec<LoopExit>,
}

#[deriving(PartialEq, Eq)]
enum LoopKind {
    /// A function, closure, or constant item (not actually a loop).
    Function,
    /// A `loop` loop.
    Loop,
    /// A `while` loop.
    While,
    /// A `for` loop.
    For,
}

#[allow(dead_code)]
struct LoopExit {
    /// The kind of loop exit that this is.
    kind: LoopExitKind,
    /// The ID of the expression that exits the loop. If this is the normal
    /// exit, this will be the ID of the block.
    expr_id: NodeId,
    /// The ID of the loop that this exits.
    loop_id: NodeId,
}

enum LoopExitKind {
    NormalExit,
    ContinueExit,
    BreakExit,
    ReturnExit,
}

pub struct LoopAnalyzer<'a> {
    analysis: LoopAnalysis,
    tcx: &'a ctxt,
}

impl<'a> LoopAnalyzer<'a> {
    pub fn new(tcx: &ctxt) -> LoopAnalyzer {
        LoopAnalyzer {
            analysis: LoopAnalysis {
                graph: Graph::new(),
                entry: NodeIndex(-1),
                scope_map: NodeMap::new(),
            },
            tcx: tcx,
        }
    }

    pub fn analyze_block(mut self, block: &Block) -> LoopAnalysis {
        let entry = self.add_node(Function, block.id, block.id, None);
        self.analysis.entry = entry;
        self.scan_block(block, entry);
        self.analysis
    }

    pub fn analyze_expr(mut self, expr: &Expr) -> LoopAnalysis {
        let entry = self.add_node(Function, expr.id, expr.id, None);
        self.analysis.entry = entry;
        self.scan_expr(expr, entry);
        self.analysis
    }

    fn scan_block(&mut self, block: &Block, parent: NodeIndex) {
        for stmt in block.stmts.iter() {
            self.scan_stmt(&**stmt, parent)
        }
        match block.expr {
            None => {}
            Some(ref expr) => self.scan_expr(&**expr, parent),
        }
    }

    fn scan_stmt(&mut self, stmt: &Stmt, parent: NodeIndex) {
        match stmt.node {
            StmtDecl(ref decl, _) => self.scan_decl(&**decl, parent),
            StmtExpr(ref expr, _) | StmtSemi(ref expr, _) => {
                self.scan_expr(&**expr, parent)
            }
            StmtMac(..) => {
                self.tcx.sess.span_bug(stmt.span, "unexpanded macro")
            }
        }
    }

    fn scan_decl(&mut self, decl: &Decl, parent: NodeIndex) {
        match decl.node {
            DeclLocal(ref local) => {
                match local.init {
                    None => {}
                    Some(ref init) => self.scan_expr(&**init, parent),
                }
            }
            DeclItem(_) => {}
        }
    }

    fn scan_expr(&mut self, expr: &Expr, parent: NodeIndex) {
        match expr.node {
            // Loops:
            ExprWhile(ref cond, ref body, _) => {
                self.scan_expr(&**cond, parent);
                let loop_index = self.add_node(While,
                                               expr.id,
                                               body.id,
                                               Some(parent));
                self.scan_block(&**body, loop_index);
            }
            ExprForLoop(_, ref head, ref body, _) => {
                self.scan_expr(&**head, parent);
                let loop_index = self.add_node(For,
                                               expr.id,
                                               body.id,
                                               Some(parent));
                self.scan_block(&**body, loop_index);
            }
            ExprLoop(ref body, _) => {
                let loop_index = self.add_node(Loop,
                                               expr.id,
                                               body.id,
                                               Some(parent));
                self.scan_block(&**body, loop_index);
            }

            // Functions:
            ExprFnBlock(_, _, ref body) |
            ExprProc(_, ref body) |
            ExprUnboxedFn(_, _, _, ref body) => {
                let fn_index = self.add_node(Function,
                                             expr.id,
                                             body.id,
                                             Some(parent));
                self.scan_block(&**body, fn_index);
            }

            // Loop exits:
            ExprBreak(label) | ExprAgain(label) => {
                let loop_index = if label.is_none() {
                    parent
                } else {
                    self.find_enclosing_loop(self.tcx
                                                 .def_map
                                                 .borrow()
                                                 .get(&expr.id)
                                                 .def_id()
                                                 .node)
                };
                let exit = LoopExit {
                    kind: match expr.node {
                        ExprAgain(_) => ContinueExit,
                        ExprBreak(_) => BreakExit,
                        _ => {
                            self.tcx.sess.bug("non-again/break loop exit")
                        }
                    },
                    expr_id: expr.id,
                    loop_id: self.analysis
                                 .graph
                                 .node_data(loop_index)
                                 .expr_id,
                };
                self.analysis.graph.mut_node_data(parent).exits.push(exit)
            }
            ExprRet(ref value) => {
                match *value {
                    None => {}
                    Some(ref value) => self.scan_expr(&**value, parent),
                }
                let parent = self.analysis.graph.mut_node_data(parent);
                parent.exits.push(LoopExit {
                    kind: ReturnExit,
                    expr_id: expr.id,
                    loop_id: self.tcx
                                 .region_maps
                                 .enclosing_function(parent.expr_id),
                })
            }

            // Others:
            ExprBox(ref lhs, ref rhs) |
            ExprBinary(_, ref lhs, ref rhs) |
            ExprAssign(ref lhs, ref rhs) |
            ExprAssignOp(_, ref lhs, ref rhs) |
            ExprIndex(ref lhs, ref rhs) |
            ExprRepeat(ref lhs, ref rhs) => {
                self.scan_expr(&**lhs, parent);
                self.scan_expr(&**rhs, parent)
            }
            ExprVec(ref subs) |
            ExprMethodCall(_, _, ref subs) |
            ExprTup(ref subs) => {
                for sub in subs.iter() {
                    self.scan_expr(&**sub, parent)
                }
            }
            ExprCall(ref callee, ref args) => {
                self.scan_expr(&**callee, parent);
                for arg in args.iter() {
                    self.scan_expr(&**arg, parent)
                }
            }
            ExprUnary(_, ref sub) |
            ExprCast(ref sub, _) |
            ExprField(ref sub, _, _) |
            ExprAddrOf(_, ref sub) |
            ExprParen(ref sub) => self.scan_expr(&**sub, parent),
            ExprIf(ref sub, ref block, ref sub_opt) => {
                self.scan_expr(&**sub, parent);
                self.scan_block(&**block, parent);
                match *sub_opt {
                    None => {}
                    Some(ref els) => self.scan_expr(&**els, parent),
                }
            }
            ExprMatch(ref sub, ref arms) => {
                self.scan_expr(&**sub, parent);
                for arm in arms.iter() {
                    match arm.guard {
                        None => {}
                        Some(ref guard) => self.scan_expr(&**guard, parent),
                    }
                    self.scan_expr(&*arm.body, parent)
                }
            }
            ExprBlock(ref block) => self.scan_block(&**block, parent),
            ExprPath(_) | ExprLit(..) => {}
            ExprInlineAsm(ref asm) => {
                for &(_, output, _) in asm.outputs.iter() {
                    self.scan_expr(&*output, parent)
                }
                for &(_, input) in asm.inputs.iter() {
                    self.scan_expr(&*input, parent)
                }
            }
            ExprStruct(_, ref fields, ref base) => {
                for field in fields.iter() {
                    self.scan_expr(&*field.expr, parent)
                }
                match *base {
                    None => {}
                    Some(ref base) => self.scan_expr(&**base, parent),
                }
            }
            ExprMac(..) => {
                self.tcx.sess.span_bug(expr.span, "unexpanded macro")
            }
        }
    }

    fn add_node(&mut self,
                kind: LoopKind,
                expr_id: NodeId,
                scope_id: NodeId,
                parent: Option<NodeIndex>)
                -> NodeIndex {
        let node_index = self.analysis.graph.add_node(LoopAnalysisNode {
            expr_id: expr_id,
            scope_id: scope_id,
            kind: kind,
            exits: vec![
                LoopExit {
                    kind: NormalExit,
                    expr_id: scope_id,
                    loop_id: scope_id,
                }
            ],
        });
        match parent {
            None => {}
            Some(parent) => {
                self.analysis.graph.add_edge(parent, node_index, ());
            }
        }
        self.analysis.scope_map.insert(scope_id, node_index);
        node_index
    }

    fn find_enclosing_loop(&self, node_id: NodeId) -> NodeIndex {
        let mut scope_id = node_id;
        loop {
            match self.analysis.scope_map.find(&scope_id) {
                Some(scope_index) => return *scope_index,
                None => {}
            }
            match self.tcx.region_maps.opt_encl_scope(scope_id) {
                Some(encl_scope_id) => scope_id = encl_scope_id,
                None => {
                    self.tcx.sess.span_bug(self.tcx.map.span(node_id),
                                           "no enclosing loop")
                }
            }
        }
    }
}

impl LoopAnalysis {
    pub fn scope_exits(&self, tcx: &ctxt, scope_id: NodeId) -> Vec<NodeId> {
        fn search(this: &LoopAnalysis,
                  tcx: &ctxt,
                  node_index: NodeIndex,
                  scope_id: NodeId,
                  accumulator: &mut Vec<NodeId>) {
            for exit in this.graph.node_data(node_index).exits.iter() {
                if tcx.region_maps.is_subscope_of(scope_id, exit.loop_id) {
                    accumulator.push(exit.expr_id)
                }
            }
            this.graph.each_outgoing_edge(node_index, |_, edge| {
                if this.graph.node_data(edge.target).kind != Function {
                    search(this, tcx, edge.target, scope_id, accumulator);
                }
                true
            });
        }

        // Find the innermost enclosing loop.
        let mut accumulator = Vec::new();
        let mut current_scope = scope_id;
        loop {
            match self.scope_map.find(&current_scope) {
                None => {}
                Some(node_index) if scope_id == current_scope => {
                    // The scope that was passed to us precisely describes a
                    // loop, so just find its exits.
                    search(self,
                           tcx,
                           *node_index,
                           current_scope,
                           &mut accumulator);
                    break;
                }
                Some(node_index) => {
                    // The scope that was passed to us describes some scope
                    // outside of a loop. Start with the main exit of that
                    // scope, and add to it all exits of the loop that leave
                    // the scope.
                    accumulator.push(scope_id);
                    self.graph.each_outgoing_edge(*node_index, |_, edge| {
                        let node_data = self.graph.node_data(edge.target);
                        if node_data.kind != Function {
                            let loop_id = node_data.expr_id;
                            if tcx.region_maps.is_subscope_of(loop_id,
                                                              scope_id) {
                                search(self,
                                       tcx,
                                       edge.target,
                                       scope_id,
                                       &mut accumulator);
                            }
                        }
                        true
                    });
                    break;
                }
            }
            match tcx.region_maps.opt_encl_scope(current_scope) {
                None => {
                    tcx.sess
                       .span_bug(tcx.map.span(scope_id),
                                 format!("didn't find any outer scope in \
                                          loop analysis (failed scope was \
                                          {})",
                                         tcx.map.node_to_string(
                                            current_scope)).as_slice())
                }
                Some(encl_scope) => current_scope = encl_scope,
            }
        }

        accumulator
    }
}

