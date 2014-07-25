// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::cfg::*;
use middle::def;
use middle::graph;
use middle::typeck;
use middle::ty;
use syntax::ast;
use syntax::ast_util;
use util::nodemap::NodeMap;

use std::gc::Gc;

struct CFGBuilder<'a> {
    tcx: &'a ty::ctxt,
    exit_map: NodeMap<CFGIndex>,
    graph: CFGGraph,
    fn_exit: CFGIndex,
    loop_scopes: Vec<LoopScope>,
}

struct LoopScope {
    loop_id: ast::NodeId,     // id of loop/while node
    continue_index: CFGIndex, // where to go on a `loop`
    break_index: CFGIndex,    // where to go on a `break
}

pub fn construct(tcx: &ty::ctxt,
                 blk: &ast::Block) -> CFG {
    let mut graph = graph::Graph::new();
    let entry = add_initial_dummy_node(&mut graph);

    // `fn_exit` is target of return exprs, which lies somewhere
    // outside input `blk`. (Distinguishing `fn_exit` and `block_exit`
    // also resolves chicken-and-egg problem that arises if you try to
    // have return exprs jump to `block_exit` during construction.)
    let fn_exit = add_initial_dummy_node(&mut graph);
    let block_exit;

    let mut cfg_builder = CFGBuilder {
        exit_map: NodeMap::new(),
        graph: graph,
        fn_exit: fn_exit,
        tcx: tcx,
        loop_scopes: Vec::new()
    };
    block_exit = cfg_builder.block(blk, entry);
    cfg_builder.add_contained_edge(block_exit, fn_exit);
    let CFGBuilder {exit_map, graph, ..} = cfg_builder;
    CFG {exit_map: exit_map,
         graph: graph,
         entry: entry,
         exit: fn_exit}
}

fn add_initial_dummy_node(g: &mut CFGGraph) -> CFGIndex {
    g.add_node(CFGNodeData { id: ast::DUMMY_NODE_ID })
}

impl<'a> CFGBuilder<'a> {
    fn block(&mut self, blk: &ast::Block, pred: CFGIndex) -> CFGIndex {
        let mut stmts_exit = pred;
        for stmt in blk.stmts.iter() {
            stmts_exit = self.stmt(stmt.clone(), stmts_exit);
        }

        let expr_exit = self.opt_expr(blk.expr.clone(), stmts_exit);

        self.add_node(blk.id, [expr_exit])
    }

    fn stmt(&mut self, stmt: Gc<ast::Stmt>, pred: CFGIndex) -> CFGIndex {
        match stmt.node {
            ast::StmtDecl(ref decl, _) => {
                self.decl(&**decl, pred)
            }

            ast::StmtExpr(ref expr, _) | ast::StmtSemi(ref expr, _) => {
                self.expr(expr.clone(), pred)
            }

            ast::StmtMac(..) => {
                self.tcx.sess.span_bug(stmt.span, "unexpanded macro");
            }
        }
    }

    fn decl(&mut self, decl: &ast::Decl, pred: CFGIndex) -> CFGIndex {
        match decl.node {
            ast::DeclLocal(ref local) => {
                let init_exit = self.opt_expr(local.init.clone(), pred);
                self.pat(&*local.pat, init_exit)
            }

            ast::DeclItem(_) => {
                pred
            }
        }
    }

    fn pat(&mut self, pat: &ast::Pat, pred: CFGIndex) -> CFGIndex {
        match pat.node {
            ast::PatIdent(_, _, None) |
            ast::PatEnum(_, None) |
            ast::PatLit(..) |
            ast::PatRange(..) |
            ast::PatWild | ast::PatWildMulti => {
                self.add_node(pat.id, [pred])
            }

            ast::PatBox(ref subpat) |
            ast::PatRegion(ref subpat) |
            ast::PatIdent(_, _, Some(ref subpat)) => {
                let subpat_exit = self.pat(&**subpat, pred);
                self.add_node(pat.id, [subpat_exit])
            }

            ast::PatEnum(_, Some(ref subpats)) |
            ast::PatTup(ref subpats) => {
                let pats_exit =
                    self.pats_all(subpats.iter().map(|p| p.clone()), pred);
                self.add_node(pat.id, [pats_exit])
            }

            ast::PatStruct(_, ref subpats, _) => {
                let pats_exit =
                    self.pats_all(subpats.iter().map(|f| f.pat.clone()), pred);
                self.add_node(pat.id, [pats_exit])
            }

            ast::PatVec(ref pre, ref vec, ref post) => {
                let pre_exit =
                    self.pats_all(pre.iter().map(|p| *p), pred);
                let vec_exit =
                    self.pats_all(vec.iter().map(|p| *p), pre_exit);
                let post_exit =
                    self.pats_all(post.iter().map(|p| *p), vec_exit);
                self.add_node(pat.id, [post_exit])
            }

            ast::PatMac(_) => {
                self.tcx.sess.span_bug(pat.span, "unexpanded macro");
            }
        }
    }

    fn pats_all<I: Iterator<Gc<ast::Pat>>>(&mut self,
                                        pats: I,
                                        pred: CFGIndex) -> CFGIndex {
        //! Handles case where all of the patterns must match.
        let mut pats = pats;
        pats.fold(pred, |pred, pat| self.pat(&*pat, pred))
    }

    fn pats_any(&mut self,
                pats: &[Gc<ast::Pat>],
                pred: CFGIndex) -> CFGIndex {
        //! Handles case where just one of the patterns must match.

        if pats.len() == 1 {
            self.pat(&*pats[0], pred)
        } else {
            let collect = self.add_dummy_node([]);
            for &pat in pats.iter() {
                let pat_exit = self.pat(&*pat, pred);
                self.add_contained_edge(pat_exit, collect);
            }
            collect
        }
    }

    fn expr(&mut self, expr: Gc<ast::Expr>, pred: CFGIndex) -> CFGIndex {
        match expr.node {
            ast::ExprBlock(ref blk) => {
                let blk_exit = self.block(&**blk, pred);
                self.add_node(expr.id, [blk_exit])
            }

            ast::ExprIf(ref cond, ref then, None) => {
                //
                //     [pred]
                //       |
                //       v 1
                //     [cond]
                //       |
                //      / \
                //     /   \
                //    v 2   *
                //  [then]  |
                //    |     |
                //    v 3   v 4
                //   [..expr..]
                //
                let cond_exit = self.expr(cond.clone(), pred);           // 1
                let then_exit = self.block(&**then, cond_exit);          // 2
                self.add_node(expr.id, [cond_exit, then_exit])           // 3,4
            }

            ast::ExprIf(ref cond, ref then, Some(ref otherwise)) => {
                //
                //     [pred]
                //       |
                //       v 1
                //     [cond]
                //       |
                //      / \
                //     /   \
                //    v 2   v 3
                //  [then][otherwise]
                //    |     |
                //    v 4   v 5
                //   [..expr..]
                //
                let cond_exit = self.expr(cond.clone(), pred);           // 1
                let then_exit = self.block(&**then, cond_exit);          // 2
                let else_exit = self.expr(otherwise.clone(), cond_exit); // 3
                self.add_node(expr.id, [then_exit, else_exit])           // 4, 5
            }

            ast::ExprWhile(ref cond, ref body) => {
                //
                //         [pred]
                //           |
                //           v 1
                //       [loopback] <--+ 5
                //           |         |
                //           v 2       |
                //   +-----[cond]      |
                //   |       |         |
                //   |       v 4       |
                //   |     [body] -----+
                //   v 3
                // [expr]
                //
                // Note that `break` and `continue` statements
                // may cause additional edges.

                // Is the condition considered part of the loop?
                let loopback = self.add_dummy_node([pred]);              // 1
                let cond_exit = self.expr(cond.clone(), loopback);       // 2
                let expr_exit = self.add_node(expr.id, [cond_exit]);     // 3
                self.loop_scopes.push(LoopScope {
                    loop_id: expr.id,
                    continue_index: loopback,
                    break_index: expr_exit
                });
                let body_exit = self.block(&**body, cond_exit);          // 4
                self.add_contained_edge(body_exit, loopback);            // 5
                self.loop_scopes.pop();
                expr_exit
            }

            ast::ExprForLoop(ref pat, ref head, ref body, _) => {
                //
                //          [pred]
                //            |
                //            v 1
                //          [head]
                //            |
                //            v 2
                //        [loopback] <--+ 7
                //            |         |
                //            v 3       |
                //   +------[cond]      |
                //   |        |         |
                //   |        v 5       |
                //   |       [pat]      |
                //   |        |         |
                //   |        v 6       |
                //   v 4    [body] -----+
                // [expr]
                //
                // Note that `break` and `continue` statements
                // may cause additional edges.

                let head = self.expr(head.clone(), pred);       // 1
                let loopback = self.add_dummy_node([head]);     // 2
                let cond = self.add_dummy_node([loopback]);     // 3
                let expr_exit = self.add_node(expr.id, [cond]); // 4
                self.loop_scopes.push(LoopScope {
                    loop_id: expr.id,
                    continue_index: loopback,
                    break_index: expr_exit,
                });
                let pat = self.pat(&**pat, cond);               // 5
                let body = self.block(&**body, pat);            // 6
                self.add_contained_edge(body, loopback);        // 7
                self.loop_scopes.pop();
                expr_exit
            }

            ast::ExprLoop(ref body, _) => {
                //
                //     [pred]
                //       |
                //       v 1
                //   [loopback] <---+
                //       |      4   |
                //       v 3        |
                //     [body] ------+
                //
                //     [expr] 2
                //
                // Note that `break` and `loop` statements
                // may cause additional edges.

                let loopback = self.add_dummy_node([pred]);              // 1
                let expr_exit = self.add_node(expr.id, []);              // 2
                self.loop_scopes.push(LoopScope {
                    loop_id: expr.id,
                    continue_index: loopback,
                    break_index: expr_exit,
                });
                let body_exit = self.block(&**body, loopback);           // 3
                self.add_contained_edge(body_exit, loopback);            // 4
                self.loop_scopes.pop();
                expr_exit
            }

            ast::ExprMatch(ref discr, ref arms) => {
                //
                //     [pred]
                //       |
                //       v 1
                //    [discr]
                //       |
                //       v 2
                //    [guard1]
                //      /  \
                //     |    \
                //     v 3  |
                //  [pat1]  |
                //     |
                //     v 4  |
                // [body1]  v
                //     |  [guard2]
                //     |    /   \
                //     | [body2] \
                //     |    |   ...
                //     |    |    |
                //     v 5  v    v
                //   [....expr....]
                //
                let discr_exit = self.expr(discr.clone(), pred);         // 1

                let expr_exit = self.add_node(expr.id, []);
                let mut guard_exit = discr_exit;
                for arm in arms.iter() {
                    guard_exit = self.opt_expr(arm.guard, guard_exit);   // 2
                    let pats_exit = self.pats_any(arm.pats.as_slice(),
                                                  guard_exit);           // 3
                    let body_exit = self.expr(arm.body.clone(), pats_exit); // 4
                    self.add_contained_edge(body_exit, expr_exit);       // 5
                }
                expr_exit
            }

            ast::ExprBinary(op, ref l, ref r) if ast_util::lazy_binop(op) => {
                //
                //     [pred]
                //       |
                //       v 1
                //      [l]
                //       |
                //      / \
                //     /   \
                //    v 2  *
                //   [r]   |
                //    |    |
                //    v 3  v 4
                //   [..exit..]
                //
                let l_exit = self.expr(l.clone(), pred);                  // 1
                let r_exit = self.expr(r.clone(), l_exit);               // 2
                self.add_node(expr.id, [l_exit, r_exit])                 // 3,4
            }

            ast::ExprRet(ref v) => {
                let v_exit = self.opt_expr(v.clone(), pred);
                let b = self.add_node(expr.id, [v_exit]);
                self.add_returning_edge(expr, b);
                self.add_node(ast::DUMMY_NODE_ID, [])
            }

            ast::ExprBreak(label) => {
                let loop_scope = self.find_scope(expr, label);
                let b = self.add_node(expr.id, [pred]);
                self.add_exiting_edge(expr, b,
                                      loop_scope, loop_scope.break_index);
                self.add_node(ast::DUMMY_NODE_ID, [])
            }

            ast::ExprAgain(label) => {
                let loop_scope = self.find_scope(expr, label);
                let a = self.add_node(expr.id, [pred]);
                self.add_exiting_edge(expr, a,
                                      loop_scope, loop_scope.continue_index);
                self.add_node(ast::DUMMY_NODE_ID, [])
            }

            ast::ExprVec(ref elems) => {
                self.straightline(expr, pred, elems.as_slice())
            }

            ast::ExprCall(ref func, ref args) => {
                self.call(expr, pred, func.clone(), args.as_slice())
            }

            ast::ExprMethodCall(_, _, ref args) => {
                self.call(expr, pred, *args.get(0), args.slice_from(1))
            }

            ast::ExprIndex(ref l, ref r) |
            ast::ExprBinary(_, ref l, ref r) if self.is_method_call(&*expr) => {
                self.call(expr, pred, l.clone(), [r.clone()])
            }

            ast::ExprUnary(_, ref e) if self.is_method_call(&*expr) => {
                self.call(expr, pred, e.clone(), [])
            }

            ast::ExprTup(ref exprs) => {
                self.straightline(expr, pred, exprs.as_slice())
            }

            ast::ExprStruct(_, ref fields, base) => {
                let base_exit = self.opt_expr(base, pred);
                let field_exprs: Vec<Gc<ast::Expr>> =
                    fields.iter().map(|f| f.expr).collect();
                self.straightline(expr, base_exit, field_exprs.as_slice())
            }

            ast::ExprRepeat(elem, count) => {
                self.straightline(expr, pred, [elem, count])
            }

            ast::ExprAssign(l, r) |
            ast::ExprAssignOp(_, l, r) => {
                self.straightline(expr, pred, [r, l])
            }

            ast::ExprIndex(l, r) |
            ast::ExprBinary(_, l, r) => { // NB: && and || handled earlier
                self.straightline(expr, pred, [l, r])
            }

            ast::ExprBox(p, e) => {
                self.straightline(expr, pred, [p, e])
            }

            ast::ExprAddrOf(_, e) |
            ast::ExprCast(e, _) |
            ast::ExprUnary(_, e) |
            ast::ExprParen(e) |
            ast::ExprVstore(e, _) |
            ast::ExprField(e, _, _) => {
                self.straightline(expr, pred, [e])
            }

            ast::ExprInlineAsm(ref inline_asm) => {
                let inputs = inline_asm.inputs.iter();
                let outputs = inline_asm.outputs.iter();
                fn extract_expr<A>(&(_, expr): &(A, Gc<ast::Expr>)) -> Gc<ast::Expr> { expr }
                let post_inputs = self.exprs(inputs.map(|a| {
                    debug!("cfg::construct InlineAsm id:{} input:{:?}", expr.id, a);
                    extract_expr(a)
                }), pred);
                let post_outputs = self.exprs(outputs.map(|a| {
                    debug!("cfg::construct InlineAsm id:{} output:{:?}", expr.id, a);
                    extract_expr(a)
                }), post_inputs);
                self.add_node(expr.id, [post_outputs])
            }

            ast::ExprMac(..) |
            ast::ExprFnBlock(..) |
            ast::ExprProc(..) |
            ast::ExprUnboxedFn(..) |
            ast::ExprLit(..) |
            ast::ExprPath(..) => {
                self.straightline(expr, pred, [])
            }
        }
    }

    fn call(&mut self,
            call_expr: Gc<ast::Expr>,
            pred: CFGIndex,
            func_or_rcvr: Gc<ast::Expr>,
            args: &[Gc<ast::Expr>]) -> CFGIndex {
        let func_or_rcvr_exit = self.expr(func_or_rcvr, pred);
        let ret = self.straightline(call_expr, func_or_rcvr_exit, args);

        let return_ty = ty::node_id_to_type(self.tcx, call_expr.id);
        let fails = ty::type_is_bot(return_ty);
        if fails {
            self.add_node(ast::DUMMY_NODE_ID, [])
        } else {
            ret
        }
    }

    fn exprs<I:Iterator<Gc<ast::Expr>>>(&mut self,
                                        mut exprs: I,
                                        pred: CFGIndex) -> CFGIndex {
        //! Constructs graph for `exprs` evaluated in order
        exprs.fold(pred, |p, e| self.expr(e, p))
    }

    fn opt_expr(&mut self,
                opt_expr: Option<Gc<ast::Expr>>,
                pred: CFGIndex) -> CFGIndex {
        //! Constructs graph for `opt_expr` evaluated, if Some

        opt_expr.iter().fold(pred, |p, &e| self.expr(e, p))
    }

    fn straightline(&mut self,
                    expr: Gc<ast::Expr>,
                    pred: CFGIndex,
                    subexprs: &[Gc<ast::Expr>]) -> CFGIndex {
        //! Handles case of an expression that evaluates `subexprs` in order

        let subexprs_exit = self.exprs(subexprs.iter().map(|&e|e), pred);
        self.add_node(expr.id, [subexprs_exit])
    }

    fn add_dummy_node(&mut self, preds: &[CFGIndex]) -> CFGIndex {
        self.add_node(ast::DUMMY_NODE_ID, preds)
    }

    fn add_node(&mut self, id: ast::NodeId, preds: &[CFGIndex]) -> CFGIndex {
        assert!(!self.exit_map.contains_key(&id));
        let node = self.graph.add_node(CFGNodeData {id: id});
        if id != ast::DUMMY_NODE_ID {
            assert!(!self.exit_map.contains_key(&id));
            self.exit_map.insert(id, node);
        }
        for &pred in preds.iter() {
            self.add_contained_edge(pred, node);
        }
        node
    }

    fn add_contained_edge(&mut self,
                          source: CFGIndex,
                          target: CFGIndex) {
        let data = CFGEdgeData {exiting_scopes: vec!() };
        self.graph.add_edge(source, target, data);
    }

    fn add_exiting_edge(&mut self,
                        from_expr: Gc<ast::Expr>,
                        from_index: CFGIndex,
                        to_loop: LoopScope,
                        to_index: CFGIndex) {
        let mut data = CFGEdgeData {exiting_scopes: vec!() };
        let mut scope_id = from_expr.id;
        while scope_id != to_loop.loop_id {

            data.exiting_scopes.push(scope_id);
            scope_id = self.tcx.region_maps.encl_scope(scope_id);
        }
        self.graph.add_edge(from_index, to_index, data);
    }

    fn add_returning_edge(&mut self,
                          _from_expr: Gc<ast::Expr>,
                          from_index: CFGIndex) {
        let mut data = CFGEdgeData {
            exiting_scopes: vec!(),
        };
        for &LoopScope { loop_id: id, .. } in self.loop_scopes.iter().rev() {
            data.exiting_scopes.push(id);
        }
        self.graph.add_edge(from_index, self.fn_exit, data);
    }

    fn find_scope(&self,
                  expr: Gc<ast::Expr>,
                  label: Option<ast::Ident>) -> LoopScope {
        match label {
            None => {
                return *self.loop_scopes.last().unwrap();
            }

            Some(_) => {
                match self.tcx.def_map.borrow().find(&expr.id) {
                    Some(&def::DefLabel(loop_id)) => {
                        for l in self.loop_scopes.iter() {
                            if l.loop_id == loop_id {
                                return *l;
                            }
                        }
                        self.tcx.sess.span_bug(
                            expr.span,
                            format!("no loop scope for id {:?}",
                                    loop_id).as_slice());
                    }

                    r => {
                        self.tcx.sess.span_bug(
                            expr.span,
                            format!("bad entry `{:?}` in def_map for label",
                                    r).as_slice());
                    }
                }
            }
        }
    }

    fn is_method_call(&self, expr: &ast::Expr) -> bool {
        let method_call = typeck::MethodCall::expr(expr.id);
        self.tcx.method_map.borrow().contains_key(&method_call)
    }
}
