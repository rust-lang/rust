use crate::cfg::*;
use crate::middle::region;
use rustc_data_structures::graph::implementation as graph;
use crate::ty::{self, TyCtxt};

use crate::hir::{self, PatKind};
use crate::hir::def_id::DefId;
use crate::hir::ptr::P;

struct CFGBuilder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    owner_def_id: DefId,
    tables: &'a ty::TypeckTables<'tcx>,
    graph: CFGGraph,
    fn_exit: CFGIndex,
    loop_scopes: Vec<LoopScope>,
    breakable_block_scopes: Vec<BlockScope>,
}

#[derive(Copy, Clone)]
struct BlockScope {
    block_expr_id: hir::ItemLocalId, // id of breakable block expr node
    break_index: CFGIndex, // where to go on `break`
}

#[derive(Copy, Clone)]
struct LoopScope {
    loop_id: hir::ItemLocalId,     // id of loop/while node
    continue_index: CFGIndex, // where to go on a `loop`
    break_index: CFGIndex,    // where to go on a `break`
}

pub fn construct(tcx: TyCtxt<'_>, body: &hir::Body) -> CFG {
    let mut graph = graph::Graph::new();
    let entry = graph.add_node(CFGNodeData::Entry);

    // `fn_exit` is target of return exprs, which lies somewhere
    // outside input `body`. (Distinguishing `fn_exit` and `body_exit`
    // also resolves chicken-and-egg problem that arises if you try to
    // have return exprs jump to `body_exit` during construction.)
    let fn_exit = graph.add_node(CFGNodeData::Exit);
    let body_exit;

    // Find the tables for this body.
    let owner_def_id = tcx.hir().body_owner_def_id(body.id());
    let tables = tcx.typeck_tables_of(owner_def_id);

    let mut cfg_builder = CFGBuilder {
        tcx,
        owner_def_id,
        tables,
        graph,
        fn_exit,
        loop_scopes: Vec::new(),
        breakable_block_scopes: Vec::new(),
    };
    body_exit = cfg_builder.expr(&body.value, entry);
    cfg_builder.add_contained_edge(body_exit, fn_exit);
    let CFGBuilder { graph, .. } = cfg_builder;
    CFG {
        owner_def_id,
        graph,
        entry,
        exit: fn_exit,
    }
}

impl<'a, 'tcx> CFGBuilder<'a, 'tcx> {
    fn block(&mut self, blk: &hir::Block, pred: CFGIndex) -> CFGIndex {
        if blk.targeted_by_break {
            let expr_exit = self.add_ast_node(blk.hir_id.local_id, &[]);

            self.breakable_block_scopes.push(BlockScope {
                block_expr_id: blk.hir_id.local_id,
                break_index: expr_exit,
            });

            let mut stmts_exit = pred;
            for stmt in &blk.stmts {
                stmts_exit = self.stmt(stmt, stmts_exit);
            }
            let blk_expr_exit = self.opt_expr(&blk.expr, stmts_exit);
            self.add_contained_edge(blk_expr_exit, expr_exit);

            self.breakable_block_scopes.pop();

            expr_exit
        } else {
            let mut stmts_exit = pred;
            for stmt in &blk.stmts {
                stmts_exit = self.stmt(stmt, stmts_exit);
            }

            let expr_exit = self.opt_expr(&blk.expr, stmts_exit);

            self.add_ast_node(blk.hir_id.local_id, &[expr_exit])
        }
    }

    fn stmt(&mut self, stmt: &hir::Stmt, pred: CFGIndex) -> CFGIndex {
        let exit = match stmt.node {
            hir::StmtKind::Local(ref local) => {
                let init_exit = self.opt_expr(&local.init, pred);
                self.pat(&local.pat, init_exit)
            }
            hir::StmtKind::Item(_) => {
                pred
            }
            hir::StmtKind::Expr(ref expr) |
            hir::StmtKind::Semi(ref expr) => {
                self.expr(&expr, pred)
            }
        };
        self.add_ast_node(stmt.hir_id.local_id, &[exit])
    }

    fn pat(&mut self, pat: &hir::Pat, pred: CFGIndex) -> CFGIndex {
        match pat.node {
            PatKind::Binding(.., None) |
            PatKind::Path(_) |
            PatKind::Lit(..) |
            PatKind::Range(..) |
            PatKind::Wild => self.add_ast_node(pat.hir_id.local_id, &[pred]),

            PatKind::Box(ref subpat) |
            PatKind::Ref(ref subpat, _) |
            PatKind::Binding(.., Some(ref subpat)) => {
                let subpat_exit = self.pat(&subpat, pred);
                self.add_ast_node(pat.hir_id.local_id, &[subpat_exit])
            }

            PatKind::TupleStruct(_, ref subpats, _) |
            PatKind::Tuple(ref subpats, _) => {
                let pats_exit = self.pats_all(subpats.iter(), pred);
                self.add_ast_node(pat.hir_id.local_id, &[pats_exit])
            }

            PatKind::Struct(_, ref subpats, _) => {
                let pats_exit = self.pats_all(subpats.iter().map(|f| &f.node.pat), pred);
                self.add_ast_node(pat.hir_id.local_id, &[pats_exit])
            }

            PatKind::Slice(ref pre, ref vec, ref post) => {
                let pre_exit = self.pats_all(pre.iter(), pred);
                let vec_exit = self.pats_all(vec.iter(), pre_exit);
                let post_exit = self.pats_all(post.iter(), vec_exit);
                self.add_ast_node(pat.hir_id.local_id, &[post_exit])
            }
        }
    }

    fn pats_all<'b, I: Iterator<Item=&'b P<hir::Pat>>>(
        &mut self,
        pats: I,
        pred: CFGIndex
    ) -> CFGIndex {
        //! Handles case where all of the patterns must match.
        pats.fold(pred, |pred, pat| self.pat(&pat, pred))
    }

    fn expr(&mut self, expr: &hir::Expr, pred: CFGIndex) -> CFGIndex {
        match expr.node {
            hir::ExprKind::Block(ref blk, _) => {
                let blk_exit = self.block(&blk, pred);
                self.add_ast_node(expr.hir_id.local_id, &[blk_exit])
            }

            hir::ExprKind::While(ref cond, ref body, _) => {
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

                let loopback = self.add_dummy_node(&[pred]);              // 1

                // Create expr_exit without pred (cond_exit)
                let expr_exit = self.add_ast_node(expr.hir_id.local_id, &[]);         // 3

                // The LoopScope needs to be on the loop_scopes stack while evaluating the
                // condition and the body of the loop (both can break out of the loop)
                self.loop_scopes.push(LoopScope {
                    loop_id: expr.hir_id.local_id,
                    continue_index: loopback,
                    break_index: expr_exit
                });

                let cond_exit = self.expr(&cond, loopback);             // 2

                // Add pred (cond_exit) to expr_exit
                self.add_contained_edge(cond_exit, expr_exit);

                let body_exit = self.block(&body, cond_exit);          // 4
                self.add_contained_edge(body_exit, loopback);            // 5
                self.loop_scopes.pop();
                expr_exit
            }

            hir::ExprKind::Loop(ref body, _, _) => {
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
                let expr_exit = self.add_ast_node(expr.hir_id.local_id, &[]);          // 2
                self.loop_scopes.push(LoopScope {
                    loop_id: expr.hir_id.local_id,
                    continue_index: loopback,
                    break_index: expr_exit,
                });
                let body_exit = self.block(&body, loopback);           // 3
                self.add_contained_edge(body_exit, loopback);            // 4
                self.loop_scopes.pop();
                expr_exit
            }

            hir::ExprKind::Match(ref discr, ref arms, _) => {
                self.match_(expr.hir_id.local_id, &discr, &arms, pred)
            }

            hir::ExprKind::Binary(op, ref l, ref r) if op.node.is_lazy() => {
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
                let l_exit = self.expr(&l, pred);                      // 1
                let r_exit = self.expr(&r, l_exit);                    // 2
                self.add_ast_node(expr.hir_id.local_id, &[l_exit, r_exit])            // 3,4
            }

            hir::ExprKind::Ret(ref v) => {
                let v_exit = self.opt_expr(v, pred);
                let b = self.add_ast_node(expr.hir_id.local_id, &[v_exit]);
                self.add_returning_edge(expr, b);
                self.add_unreachable_node()
            }

            hir::ExprKind::Break(destination, ref opt_expr) => {
                let v = self.opt_expr(opt_expr, pred);
                let (target_scope, break_dest) =
                    self.find_scope_edge(expr, destination, ScopeCfKind::Break);
                let b = self.add_ast_node(expr.hir_id.local_id, &[v]);
                self.add_exiting_edge(expr, b, target_scope, break_dest);
                self.add_unreachable_node()
            }

            hir::ExprKind::Continue(destination) => {
                let (target_scope, cont_dest) =
                    self.find_scope_edge(expr, destination, ScopeCfKind::Continue);
                let a = self.add_ast_node(expr.hir_id.local_id, &[pred]);
                self.add_exiting_edge(expr, a, target_scope, cont_dest);
                self.add_unreachable_node()
            }

            hir::ExprKind::Array(ref elems) => {
                self.straightline(expr, pred, elems.iter().map(|e| &*e))
            }

            hir::ExprKind::Call(ref func, ref args) => {
                self.call(expr, pred, &func, args.iter().map(|e| &*e))
            }

            hir::ExprKind::MethodCall(.., ref args) => {
                self.call(expr, pred, &args[0], args[1..].iter().map(|e| &*e))
            }

            hir::ExprKind::Index(ref l, ref r) |
            hir::ExprKind::Binary(_, ref l, ref r) if self.tables.is_method_call(expr) => {
                self.call(expr, pred, &l, Some(&**r).into_iter())
            }

            hir::ExprKind::Unary(_, ref e) if self.tables.is_method_call(expr) => {
                self.call(expr, pred, &e, None::<hir::Expr>.iter())
            }

            hir::ExprKind::Tup(ref exprs) => {
                self.straightline(expr, pred, exprs.iter().map(|e| &*e))
            }

            hir::ExprKind::Struct(_, ref fields, ref base) => {
                let field_cfg = self.straightline(expr, pred, fields.iter().map(|f| &*f.expr));
                self.opt_expr(base, field_cfg)
            }

            hir::ExprKind::Assign(ref l, ref r) |
            hir::ExprKind::AssignOp(_, ref l, ref r) => {
                self.straightline(expr, pred, [r, l].iter().map(|&e| &**e))
            }

            hir::ExprKind::Index(ref l, ref r) |
            hir::ExprKind::Binary(_, ref l, ref r) => { // N.B., && and || handled earlier
                self.straightline(expr, pred, [l, r].iter().map(|&e| &**e))
            }

            hir::ExprKind::Box(ref e) |
            hir::ExprKind::AddrOf(_, ref e) |
            hir::ExprKind::Cast(ref e, _) |
            hir::ExprKind::Type(ref e, _) |
            hir::ExprKind::DropTemps(ref e) |
            hir::ExprKind::Unary(_, ref e) |
            hir::ExprKind::Field(ref e, _) |
            hir::ExprKind::Yield(ref e, _) |
            hir::ExprKind::Repeat(ref e, _) => {
                self.straightline(expr, pred, Some(&**e).into_iter())
            }

            hir::ExprKind::InlineAsm(_, ref outputs, ref inputs) => {
                let post_outputs = self.exprs(outputs.iter().map(|e| &*e), pred);
                let post_inputs = self.exprs(inputs.iter().map(|e| &*e), post_outputs);
                self.add_ast_node(expr.hir_id.local_id, &[post_inputs])
            }

            hir::ExprKind::Closure(..) |
            hir::ExprKind::Lit(..) |
            hir::ExprKind::Path(_) |
            hir::ExprKind::Err => {
                self.straightline(expr, pred, None::<hir::Expr>.iter())
            }
        }
    }

    fn call<'b, I: Iterator<Item=&'b hir::Expr>>(&mut self,
            call_expr: &hir::Expr,
            pred: CFGIndex,
            func_or_rcvr: &hir::Expr,
            args: I) -> CFGIndex {
        let func_or_rcvr_exit = self.expr(func_or_rcvr, pred);
        let ret = self.straightline(call_expr, func_or_rcvr_exit, args);
        let m = self.tcx.hir().get_module_parent(call_expr.hir_id);
        if self.tcx.is_ty_uninhabited_from(m, self.tables.expr_ty(call_expr)) {
            self.add_unreachable_node()
        } else {
            ret
        }
    }

    fn exprs<'b, I: Iterator<Item=&'b hir::Expr>>(&mut self,
                                             exprs: I,
                                             pred: CFGIndex) -> CFGIndex {
        //! Constructs graph for `exprs` evaluated in order
        exprs.fold(pred, |p, e| self.expr(e, p))
    }

    fn opt_expr(&mut self,
                opt_expr: &Option<P<hir::Expr>>,
                pred: CFGIndex) -> CFGIndex {
        //! Constructs graph for `opt_expr` evaluated, if Some
        opt_expr.iter().fold(pred, |p, e| self.expr(&e, p))
    }

    fn straightline<'b, I: Iterator<Item=&'b hir::Expr>>(&mut self,
                    expr: &hir::Expr,
                    pred: CFGIndex,
                    subexprs: I) -> CFGIndex {
        //! Handles case of an expression that evaluates `subexprs` in order

        let subexprs_exit = self.exprs(subexprs, pred);
        self.add_ast_node(expr.hir_id.local_id, &[subexprs_exit])
    }

    fn match_(&mut self, id: hir::ItemLocalId, discr: &hir::Expr,
              arms: &[hir::Arm], pred: CFGIndex) -> CFGIndex {
        // The CFG for match expression is quite complex, so no ASCII
        // art for it (yet).
        //
        // The CFG generated below matches roughly what MIR contains.
        // Each pattern and guard is visited in parallel, with
        // arms containing multiple patterns generating multiple nodes
        // for the same guard expression. The guard expressions chain
        // into each other from top to bottom, with a specific
        // exception to allow some additional valid programs
        // (explained below). MIR differs slightly in that the
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

        for arm in arms {
            // Add an exit node for when we've visited all the
            // patterns and the guard (if there is one) in the arm.
            let bindings_exit = self.add_dummy_node(&[]);

            for pat in &arm.pats {
                // Visit the pattern, coming from the discriminant exit
                let mut pat_exit = self.pat(&pat, discr_exit);

                // If there is a guard expression, handle it here
                if let Some(ref guard) = arm.guard {
                    // Add a dummy node for the previous guard
                    // expression to target
                    let guard_start = self.add_dummy_node(&[pat_exit]);
                    // Visit the guard expression
                    let guard_exit = match guard {
                        hir::Guard::If(ref e) => self.expr(e, guard_start),
                    };
                    // #47295: We used to have very special case code
                    // here for when a pair of arms are both formed
                    // solely from constants, and if so, not add these
                    // edges.  But this was not actually sound without
                    // other constraints that we stopped enforcing at
                    // some point.
                    while let Some(prev) = prev_guards.pop() {
                        self.add_contained_edge(prev, guard_start);
                    }

                    // Push the guard onto the list of previous guards
                    prev_guards.push(guard_exit);

                    // Update the exit node for the pattern
                    pat_exit = guard_exit;
                }

                // Add an edge from the exit of this pattern to the
                // exit of the arm
                self.add_contained_edge(pat_exit, bindings_exit);
            }

            // Visit the body of this arm
            let body_exit = self.expr(&arm.body, bindings_exit);

            let arm_exit = self.add_ast_node(arm.hir_id.local_id, &[body_exit]);

            // Link the body to the exit of the expression
            self.add_contained_edge(arm_exit, expr_exit);
        }

        expr_exit
    }

    fn add_dummy_node(&mut self, preds: &[CFGIndex]) -> CFGIndex {
        self.add_node(CFGNodeData::Dummy, preds)
    }

    fn add_ast_node(&mut self, id: hir::ItemLocalId, preds: &[CFGIndex]) -> CFGIndex {
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
        let data = CFGEdgeData {exiting_scopes: vec![] };
        self.graph.add_edge(source, target, data);
    }

    fn add_exiting_edge(&mut self,
                        from_expr: &hir::Expr,
                        from_index: CFGIndex,
                        target_scope: region::Scope,
                        to_index: CFGIndex) {
        let mut data = CFGEdgeData { exiting_scopes: vec![] };
        let mut scope = region::Scope {
            id: from_expr.hir_id.local_id,
            data: region::ScopeData::Node
        };
        let region_scope_tree = self.tcx.region_scope_tree(self.owner_def_id);
        while scope != target_scope {
            data.exiting_scopes.push(scope.item_local_id());
            scope = region_scope_tree.encl_scope(scope);
        }
        self.graph.add_edge(from_index, to_index, data);
    }

    fn add_returning_edge(&mut self,
                          _from_expr: &hir::Expr,
                          from_index: CFGIndex) {
        let data = CFGEdgeData {
            exiting_scopes: self.loop_scopes.iter()
                                            .rev()
                                            .map(|&LoopScope { loop_id: id, .. }| id)
                                            .collect()
        };
        self.graph.add_edge(from_index, self.fn_exit, data);
    }

    fn find_scope_edge(&self,
                  expr: &hir::Expr,
                  destination: hir::Destination,
                  scope_cf_kind: ScopeCfKind) -> (region::Scope, CFGIndex) {

        match destination.target_id {
            Ok(loop_id) => {
                for b in &self.breakable_block_scopes {
                    if b.block_expr_id == loop_id.local_id {
                        let scope = region::Scope {
                            id: loop_id.local_id,
                            data: region::ScopeData::Node
                        };
                        return (scope, match scope_cf_kind {
                            ScopeCfKind::Break => b.break_index,
                            ScopeCfKind::Continue => bug!("can't continue to block"),
                        });
                    }
                }
                for l in &self.loop_scopes {
                    if l.loop_id == loop_id.local_id {
                        let scope = region::Scope {
                            id: loop_id.local_id,
                            data: region::ScopeData::Node
                        };
                        return (scope, match scope_cf_kind {
                            ScopeCfKind::Break => l.break_index,
                            ScopeCfKind::Continue => l.continue_index,
                        });
                    }
                }
                span_bug!(expr.span, "no scope for id {}", loop_id);
            }
            Err(err) => span_bug!(expr.span, "scope error: {}",  err),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum ScopeCfKind {
    Break,
    Continue,
}
