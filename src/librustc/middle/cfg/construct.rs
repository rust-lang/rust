// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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
use middle::pat_util;
use middle::region::CodeExtent;
use middle::ty;
use syntax::ast;
use syntax::ast_util;
use syntax::ptr::P;

struct CFGBuilder<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    graph: CFGGraph,
    fn_exit: CFGIndex,
    loop_scopes: Vec<LoopScope>,
}

#[derive(Copy, Clone)]
struct LoopScope {
    loop_id: ast::NodeId,     // id of loop/while node
    continue_index: CFGIndex, // where to go on a `loop`
    break_index: CFGIndex,    // where to go on a `break
}

pub fn construct(tcx: &ty::ctxt,
                 blk: &ast::Block) -> CFG {
    let mut graph = graph::Graph::new();
    let entry = graph.add_node(CFGNodeData::Entry);

    // `fn_exit` is target of return exprs, which lies somewhere
    // outside input `blk`. (Distinguishing `fn_exit` and `block_exit`
    // also resolves chicken-and-egg problem that arises if you try to
    // have return exprs jump to `block_exit` during construction.)
    let fn_exit = graph.add_node(CFGNodeData::Exit);
    let block_exit;

    let mut cfg_builder = CFGBuilder {
        graph: graph,
        fn_exit: fn_exit,
        tcx: tcx,
        loop_scopes: Vec::new()
    };
    block_exit = cfg_builder.block(blk, entry);
    cfg_builder.add_contained_edge(block_exit, fn_exit);
    let CFGBuilder {graph, ..} = cfg_builder;
    CFG {graph: graph,
         entry: entry,
         exit: fn_exit}
}

impl<'a, 'tcx> CFGBuilder<'a, 'tcx> {
    fn block(&mut self, blk: &ast::Block, pred: CFGIndex) -> CFGIndex {
        let mut stmts_exit = pred;
        for stmt in &blk.stmts {
            stmts_exit = self.stmt(&**stmt, stmts_exit);
        }

        let expr_exit = self.opt_expr(&blk.expr, stmts_exit);

        self.add_ast_node(blk.id, &[expr_exit])
    }

    fn stmt(&mut self, stmt: &ast::Stmt, pred: CFGIndex) -> CFGIndex {
        match stmt.node {
            ast::StmtDecl(ref decl, id) => {
                let exit = self.decl(&**decl, pred);
                self.add_ast_node(id, &[exit])
            }

            ast::StmtExpr(ref expr, id) | ast::StmtSemi(ref expr, id) => {
                let exit = self.expr(&**expr, pred);
                self.add_ast_node(id, &[exit])
            }

            ast::StmtMac(..) => {
                self.tcx.sess.span_bug(stmt.span, "unexpanded macro");
            }
        }
    }

    fn decl(&mut self, decl: &ast::Decl, pred: CFGIndex) -> CFGIndex {
        match decl.node {
            ast::DeclLocal(ref local) => {
                let init_exit = self.opt_expr(&local.init, pred);
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
            ast::PatWild(_) => {
                self.add_ast_node(pat.id, &[pred])
            }

            ast::PatBox(ref subpat) |
            ast::PatRegion(ref subpat, _) |
            ast::PatIdent(_, _, Some(ref subpat)) => {
                let subpat_exit = self.pat(&**subpat, pred);
                self.add_ast_node(pat.id, &[subpat_exit])
            }

            ast::PatEnum(_, Some(ref subpats)) |
            ast::PatTup(ref subpats) => {
                let pats_exit = self.pats_all(subpats.iter(), pred);
                self.add_ast_node(pat.id, &[pats_exit])
            }

            ast::PatStruct(_, ref subpats, _) => {
                let pats_exit =
                    self.pats_all(subpats.iter().map(|f| &f.node.pat), pred);
                self.add_ast_node(pat.id, &[pats_exit])
            }

            ast::PatVec(ref pre, ref vec, ref post) => {
                let pre_exit = self.pats_all(pre.iter(), pred);
                let vec_exit = self.pats_all(vec.iter(), pre_exit);
                let post_exit = self.pats_all(post.iter(), vec_exit);
                self.add_ast_node(pat.id, &[post_exit])
            }

            ast::PatMac(_) => {
                self.tcx.sess.span_bug(pat.span, "unexpanded macro");
            }
        }
    }

    fn pats_all<'b, I: Iterator<Item=&'b P<ast::Pat>>>(&mut self,
                                          pats: I,
                                          pred: CFGIndex) -> CFGIndex {
        //! Handles case where all of the patterns must match.
        pats.fold(pred, |pred, pat| self.pat(&**pat, pred))
    }

    fn expr(&mut self, expr: &ast::Expr, pred: CFGIndex) -> CFGIndex {
        match expr.node {
            ast::ExprBlock(ref blk) => {
                let blk_exit = self.block(&**blk, pred);
                self.add_ast_node(expr.id, &[blk_exit])
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
                let cond_exit = self.expr(&**cond, pred);                // 1
                let then_exit = self.block(&**then, cond_exit);          // 2
                self.add_ast_node(expr.id, &[cond_exit, then_exit])      // 3,4
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
                let cond_exit = self.expr(&**cond, pred);                // 1
                let then_exit = self.block(&**then, cond_exit);          // 2
                let else_exit = self.expr(&**otherwise, cond_exit);      // 3
                self.add_ast_node(expr.id, &[then_exit, else_exit])      // 4, 5
            }

            ast::ExprIfLet(..) => {
                self.tcx.sess.span_bug(expr.span, "non-desugared ExprIfLet");
            }

            ast::ExprWhile(ref cond, ref body, _) => {
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
                let loopback = self.add_dummy_node(&[pred]);              // 1
                let cond_exit = self.expr(&**cond, loopback);             // 2
                let expr_exit = self.add_ast_node(expr.id, &[cond_exit]); // 3
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

            ast::ExprWhileLet(..) => {
                self.tcx.sess.span_bug(expr.span, "non-desugared ExprWhileLet");
            }

            ast::ExprForLoop(..) => {
                self.tcx.sess.span_bug(expr.span, "non-desugared ExprForLoop");
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

                let loopback = self.add_dummy_node(&[pred]);              // 1
                let expr_exit = self.add_ast_node(expr.id, &[]);          // 2
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

            ast::ExprMatch(ref discr, ref arms, _) => {
                self.match_(expr.id, &discr, &arms, pred)
            }

            ast::ExprBinary(op, ref l, ref r) if ast_util::lazy_binop(op.node) => {
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
                let l_exit = self.expr(&**l, pred);                      // 1
                let r_exit = self.expr(&**r, l_exit);                    // 2
                self.add_ast_node(expr.id, &[l_exit, r_exit])            // 3,4
            }

            ast::ExprRet(ref v) => {
                let v_exit = self.opt_expr(v, pred);
                let b = self.add_ast_node(expr.id, &[v_exit]);
                self.add_returning_edge(expr, b);
                self.add_unreachable_node()
            }

            ast::ExprBreak(label) => {
                let loop_scope = self.find_scope(expr, label);
                let b = self.add_ast_node(expr.id, &[pred]);
                self.add_exiting_edge(expr, b,
                                      loop_scope, loop_scope.break_index);
                self.add_unreachable_node()
            }

            ast::ExprAgain(label) => {
                let loop_scope = self.find_scope(expr, label);
                let a = self.add_ast_node(expr.id, &[pred]);
                self.add_exiting_edge(expr, a,
                                      loop_scope, loop_scope.continue_index);
                self.add_unreachable_node()
            }

            ast::ExprVec(ref elems) => {
                self.straightline(expr, pred, elems.iter().map(|e| &**e))
            }

            ast::ExprCall(ref func, ref args) => {
                self.call(expr, pred, &**func, args.iter().map(|e| &**e))
            }

            ast::ExprMethodCall(_, _, ref args) => {
                self.call(expr, pred, &*args[0], args[1..].iter().map(|e| &**e))
            }

            ast::ExprIndex(ref l, ref r) |
            ast::ExprBinary(_, ref l, ref r) if self.is_method_call(expr) => {
                self.call(expr, pred, &**l, Some(&**r).into_iter())
            }

            ast::ExprRange(ref start, ref end) => {
                let fields = start.as_ref().map(|e| &**e).into_iter()
                    .chain(end.as_ref().map(|e| &**e).into_iter());
                self.straightline(expr, pred, fields)
            }

            ast::ExprUnary(_, ref e) if self.is_method_call(expr) => {
                self.call(expr, pred, &**e, None::<ast::Expr>.iter())
            }

            ast::ExprTup(ref exprs) => {
                self.straightline(expr, pred, exprs.iter().map(|e| &**e))
            }

            ast::ExprStruct(_, ref fields, ref base) => {
                let field_cfg = self.straightline(expr, pred, fields.iter().map(|f| &*f.expr));
                self.opt_expr(base, field_cfg)
            }

            ast::ExprRepeat(ref elem, ref count) => {
                self.straightline(expr, pred, [elem, count].iter().map(|&e| &**e))
            }

            ast::ExprAssign(ref l, ref r) |
            ast::ExprAssignOp(_, ref l, ref r) => {
                self.straightline(expr, pred, [r, l].iter().map(|&e| &**e))
            }

            ast::ExprBox(Some(ref l), ref r) |
            ast::ExprIndex(ref l, ref r) |
            ast::ExprBinary(_, ref l, ref r) => { // NB: && and || handled earlier
                self.straightline(expr, pred, [l, r].iter().map(|&e| &**e))
            }

            ast::ExprBox(None, ref e) |
            ast::ExprAddrOf(_, ref e) |
            ast::ExprCast(ref e, _) |
            ast::ExprUnary(_, ref e) |
            ast::ExprParen(ref e) |
            ast::ExprField(ref e, _) |
            ast::ExprTupField(ref e, _) => {
                self.straightline(expr, pred, Some(&**e).into_iter())
            }

            ast::ExprInlineAsm(ref inline_asm) => {
                let inputs = inline_asm.inputs.iter();
                let outputs = inline_asm.outputs.iter();
                let post_inputs = self.exprs(inputs.map(|a| {
                    debug!("cfg::construct InlineAsm id:{} input:{:?}", expr.id, a);
                    let &(_, ref expr) = a;
                    &**expr
                }), pred);
                let post_outputs = self.exprs(outputs.map(|a| {
                    debug!("cfg::construct InlineAsm id:{} output:{:?}", expr.id, a);
                    let &(_, ref expr, _) = a;
                    &**expr
                }), post_inputs);
                self.add_ast_node(expr.id, &[post_outputs])
            }

            ast::ExprMac(..) |
            ast::ExprClosure(..) |
            ast::ExprLit(..) |
            ast::ExprPath(..) => {
                self.straightline(expr, pred, None::<ast::Expr>.iter())
            }
        }
    }

    fn call<'b, I: Iterator<Item=&'b ast::Expr>>(&mut self,
            call_expr: &ast::Expr,
            pred: CFGIndex,
            func_or_rcvr: &ast::Expr,
            args: I) -> CFGIndex {
        let method_call = ty::MethodCall::expr(call_expr.id);
        let return_ty = ty::ty_fn_ret(match self.tcx.method_map.borrow().get(&method_call) {
            Some(method) => method.ty,
            None => ty::expr_ty_adjusted(self.tcx, func_or_rcvr)
        });

        let func_or_rcvr_exit = self.expr(func_or_rcvr, pred);
        let ret = self.straightline(call_expr, func_or_rcvr_exit, args);
        if return_ty.diverges() {
            self.add_unreachable_node()
        } else {
            ret
        }
    }

    fn exprs<'b, I: Iterator<Item=&'b ast::Expr>>(&mut self,
                                             exprs: I,
                                             pred: CFGIndex) -> CFGIndex {
        //! Constructs graph for `exprs` evaluated in order
        exprs.fold(pred, |p, e| self.expr(e, p))
    }

    fn opt_expr(&mut self,
                opt_expr: &Option<P<ast::Expr>>,
                pred: CFGIndex) -> CFGIndex {
        //! Constructs graph for `opt_expr` evaluated, if Some
        opt_expr.iter().fold(pred, |p, e| self.expr(&**e, p))
    }

    fn straightline<'b, I: Iterator<Item=&'b ast::Expr>>(&mut self,
                    expr: &ast::Expr,
                    pred: CFGIndex,
                    subexprs: I) -> CFGIndex {
        //! Handles case of an expression that evaluates `subexprs` in order

        let subexprs_exit = self.exprs(subexprs, pred);
        self.add_ast_node(expr.id, &[subexprs_exit])
    }

    fn match_(&mut self, id: ast::NodeId, discr: &ast::Expr,
              arms: &[ast::Arm], pred: CFGIndex) -> CFGIndex {
        // The CFG for match expression is quite complex, so no ASCII
        // art for it (yet).
        //
        // The CFG generated below matches roughly what trans puts
        // out. Each pattern and guard is visited in parallel, with
        // arms containing multiple patterns generating multiple nodes
        // for the same guard expression. The guard expressions chain
        // into each other from top to bottom, with a specific
        // exception to allow some additional valid programs
        // (explained below). Trans differs slightly in that the
        // pattern matching may continue after a guard but the visible
        // behaviour should be the same.
        //
        // What is going on is explained in further comments.

        // Visit the discriminant expression
        let discr_exit = self.expr(discr, pred);

        // Add a node for the exit of the match expression as a whole.
        let expr_exit = self.add_ast_node(id, &[]);

        // Keep track of the previous guard expressions
        let mut prev_guards = Vec::new();
        // Track if the previous pattern contained bindings or wildcards
        let mut prev_has_bindings = false;

        for arm in arms {
            // Add an exit node for when we've visited all the
            // patterns and the guard (if there is one) in the arm.
            let arm_exit = self.add_dummy_node(&[]);

            for pat in &arm.pats {
                // Visit the pattern, coming from the discriminant exit
                let mut pat_exit = self.pat(&**pat, discr_exit);

                // If there is a guard expression, handle it here
                if let Some(ref guard) = arm.guard {
                    // Add a dummy node for the previous guard
                    // expression to target
                    let guard_start = self.add_dummy_node(&[pat_exit]);
                    // Visit the guard expression
                    let guard_exit = self.expr(&**guard, guard_start);

                    let this_has_bindings = pat_util::pat_contains_bindings_or_wild(
                        &self.tcx.def_map, &**pat);

                    // If both this pattern and the previous pattern
                    // were free of bindings, they must consist only
                    // of "constant" patterns. Note we cannot match an
                    // all-constant pattern, fail the guard, and then
                    // match *another* all-constant pattern. This is
                    // because if the previous pattern matches, then
                    // we *cannot* match this one, unless all the
                    // constants are the same (which is rejected by
                    // `check_match`).
                    //
                    // We can use this to be smarter about the flow
                    // along guards. If the previous pattern matched,
                    // then we know we will not visit the guard in
                    // this one (whether or not the guard succeeded),
                    // if the previous pattern failed, then we know
                    // the guard for that pattern will not have been
                    // visited. Thus, it is not possible to visit both
                    // the previous guard and the current one when
                    // both patterns consist only of constant
                    // sub-patterns.
                    //
                    // However, if the above does not hold, then all
                    // previous guards need to be wired to visit the
                    // current guard pattern.
                    if prev_has_bindings || this_has_bindings {
                        while let Some(prev) = prev_guards.pop() {
                            self.add_contained_edge(prev, guard_start);
                        }
                    }

                    prev_has_bindings = this_has_bindings;

                    // Push the guard onto the list of previous guards
                    prev_guards.push(guard_exit);

                    // Update the exit node for the pattern
                    pat_exit = guard_exit;
                }

                // Add an edge from the exit of this pattern to the
                // exit of the arm
                self.add_contained_edge(pat_exit, arm_exit);
            }

            // Visit the body of this arm
            let body_exit = self.expr(&arm.body, arm_exit);

            // Link the body to the exit of the expression
            self.add_contained_edge(body_exit, expr_exit);
        }

        expr_exit
    }

    fn add_dummy_node(&mut self, preds: &[CFGIndex]) -> CFGIndex {
        self.add_node(CFGNodeData::Dummy, preds)
    }

    fn add_ast_node(&mut self, id: ast::NodeId, preds: &[CFGIndex]) -> CFGIndex {
        assert!(id != ast::DUMMY_NODE_ID);
        self.add_node(CFGNodeData::AST(id), preds)
    }

    fn add_unreachable_node(&mut self) -> CFGIndex {
        self.add_node(CFGNodeData::Unreachable, &[])
    }

    fn add_node(&mut self, data: CFGNodeData, preds: &[CFGIndex]) -> CFGIndex {
        let node = self.graph.add_node(data);
        for &pred in preds {
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
                        from_expr: &ast::Expr,
                        from_index: CFGIndex,
                        to_loop: LoopScope,
                        to_index: CFGIndex) {
        let mut data = CFGEdgeData {exiting_scopes: vec!() };
        let mut scope = CodeExtent::from_node_id(from_expr.id);
        let target_scope = CodeExtent::from_node_id(to_loop.loop_id);
        while scope != target_scope {

            data.exiting_scopes.push(scope.node_id());
            scope = self.tcx.region_maps.encl_scope(scope);
        }
        self.graph.add_edge(from_index, to_index, data);
    }

    fn add_returning_edge(&mut self,
                          _from_expr: &ast::Expr,
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
                  expr: &ast::Expr,
                  label: Option<ast::Ident>) -> LoopScope {
        if label.is_none() {
            return *self.loop_scopes.last().unwrap();
        }

        match self.tcx.def_map.borrow().get(&expr.id).map(|d| d.full_def()) {
            Some(def::DefLabel(loop_id)) => {
                for l in &self.loop_scopes {
                    if l.loop_id == loop_id {
                        return *l;
                    }
                }
                self.tcx.sess.span_bug(expr.span,
                    &format!("no loop scope for id {}", loop_id));
            }

            r => {
                self.tcx.sess.span_bug(expr.span,
                    &format!("bad entry `{:?}` in def_map for label", r));
            }
        }
    }

    fn is_method_call(&self, expr: &ast::Expr) -> bool {
        let method_call = ty::MethodCall::expr(expr.id);
        self.tcx.method_map.borrow().contains_key(&method_call)
    }
}
