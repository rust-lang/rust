// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A classic liveness analysis based on dataflow over the AST.  Computes,
//! for each local variable in a function, whether that variable is live
//! at a given point.  Program execution points are identified by their
//! id.
//!
//! # Basic idea
//!
//! The basic model is that each local variable is assigned an index.  We
//! represent sets of local variables using a vector indexed by this
//! index.  The value in the vector is either 0, indicating the variable
//! is dead, or the id of an expression that uses the variable.
//!
//! We conceptually walk over the AST in reverse execution order.  If we
//! find a use of a variable, we add it to the set of live variables.  If
//! we find an assignment to a variable, we remove it from the set of live
//! variables.  When we have to merge two flows, we take the union of
//! those two flows---if the variable is live on both paths, we simply
//! pick one id.  In the event of loops, we continue doing this until a
//! fixed point is reached.
//!
//! ## Checking initialization
//!
//! At the function entry point, all variables must be dead.  If this is
//! not the case, we can report an error using the id found in the set of
//! live variables, which identifies a use of the variable which is not
//! dominated by an assignment.
//!
//! ## Checking moves
//!
//! After each explicit move, the variable must be dead.
//!
//! ## Computing last uses
//!
//! Any use of the variable where the variable is dead afterwards is a
//! last use.
//!
//! # Implementation details
//!
//! The actual implementation contains two (nested) walks over the AST.
//! The outer walk has the job of building up the ir_maps instance for the
//! enclosing function.  On the way down the tree, it identifies those AST
//! nodes and variable IDs that will be needed for the liveness analysis
//! and assigns them contiguous IDs.  The liveness id for an AST node is
//! called a `live_node` (it's a newtype'd usize) and the id for a variable
//! is called a `variable` (another newtype'd usize).
//!
//! On the way back up the tree, as we are about to exit from a function
//! declaration we allocate a `liveness` instance.  Now that we know
//! precisely how many nodes and variables we need, we can allocate all
//! the various arrays that we will need to precisely the right size.  We then
//! perform the actual propagation on the `liveness` instance.
//!
//! This propagation is encoded in the various `propagate_through_*()`
//! methods.  It effectively does a reverse walk of the AST; whenever we
//! reach a loop node, we iterate until a fixed point is reached.
//!
//! ## The `Users` struct
//!
//! At each live node `N`, we track three pieces of information for each
//! variable `V` (these are encapsulated in the `Users` struct):
//!
//! - `reader`: the `LiveNode` ID of some node which will read the value
//!    that `V` holds on entry to `N`.  Formally: a node `M` such
//!    that there exists a path `P` from `N` to `M` where `P` does not
//!    write `V`.  If the `reader` is `invalid_node()`, then the current
//!    value will never be read (the variable is dead, essentially).
//!
//! - `writer`: the `LiveNode` ID of some node which will write the
//!    variable `V` and which is reachable from `N`.  Formally: a node `M`
//!    such that there exists a path `P` from `N` to `M` and `M` writes
//!    `V`.  If the `writer` is `invalid_node()`, then there is no writer
//!    of `V` that follows `N`.
//!
//! - `used`: a boolean value indicating whether `V` is *used*.  We
//!   distinguish a *read* from a *use* in that a *use* is some read that
//!   is not just used to generate a new value.  For example, `x += 1` is
//!   a read but not a use.  This is used to generate better warnings.
//!
//! ## Special Variables
//!
//! We generate various special variables for various, well, special purposes.
//! These are described in the `specials` struct:
//!
//! - `exit_ln`: a live node that is generated to represent every 'exit' from
//!   the function, whether it be by explicit return, panic, or other means.
//!
//! - `fallthrough_ln`: a live node that represents a fallthrough
//!
//! - `no_ret_var`: a synthetic variable that is only 'read' from, the
//!   fallthrough node.  This allows us to detect functions where we fail
//!   to return explicitly.
//! - `clean_exit_var`: a synthetic variable that is only 'read' from the
//!   fallthrough node.  It is only live if the function could converge
//!   via means other than an explicit `return` expression. That is, it is
//!   only dead if the end of the function's block can never be reached.
//!   It is the responsibility of typeck to ensure that there are no
//!   `return` expressions in a function declared as diverging.
use self::LoopKind::*;
use self::LiveNodeKind::*;
use self::VarKind::*;

use middle::def::*;
use middle::pat_util;
use middle::ty;
use lint;
use util::nodemap::NodeMap;

use std::{fmt, usize};
use std::io::prelude::*;
use std::io;
use std::rc::Rc;
use syntax::ast::{self, NodeId};
use syntax::codemap::{BytePos, original_sp, Span};
use syntax::parse::token::special_idents;
use syntax::ptr::P;

use rustc_front::hir::Expr;
use rustc_front::hir;
use rustc_front::print::pprust::{expr_to_string, block_to_string};
use rustc_front::intravisit::{self, Visitor, FnKind};

/// For use with `propagate_through_loop`.
enum LoopKind<'a> {
    /// An endless `loop` loop.
    LoopLoop,
    /// A `while` loop, with the given expression as condition.
    WhileLoop(&'a Expr),
}

#[derive(Copy, Clone, PartialEq)]
struct Variable(usize);

#[derive(Copy, PartialEq)]
struct LiveNode(usize);

impl Variable {
    fn get(&self) -> usize { let Variable(v) = *self; v }
}

impl LiveNode {
    fn get(&self) -> usize { let LiveNode(v) = *self; v }
}

impl Clone for LiveNode {
    fn clone(&self) -> LiveNode {
        LiveNode(self.get())
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum LiveNodeKind {
    FreeVarNode(Span),
    ExprNode(Span),
    VarDefNode(Span),
    ExitNode
}

fn live_node_kind_to_string(lnk: LiveNodeKind, cx: &ty::ctxt) -> String {
    let cm = cx.sess.codemap();
    match lnk {
        FreeVarNode(s) => {
            format!("Free var node [{}]", cm.span_to_string(s))
        }
        ExprNode(s) => {
            format!("Expr node [{}]", cm.span_to_string(s))
        }
        VarDefNode(s) => {
            format!("Var def node [{}]", cm.span_to_string(s))
        }
        ExitNode => "Exit node".to_string(),
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for IrMaps<'a, 'tcx> {
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v hir::FnDecl,
                b: &'v hir::Block, s: Span, id: NodeId) {
        visit_fn(self, fk, fd, b, s, id);
    }
    fn visit_local(&mut self, l: &hir::Local) { visit_local(self, l); }
    fn visit_expr(&mut self, ex: &Expr) { visit_expr(self, ex); }
    fn visit_arm(&mut self, a: &hir::Arm) { visit_arm(self, a); }
}

pub fn check_crate(tcx: &ty::ctxt) {
    tcx.map.krate().visit_all_items(&mut IrMaps::new(tcx));
    tcx.sess.abort_if_errors();
}

impl fmt::Debug for LiveNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ln({})", self.get())
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "v({})", self.get())
    }
}

// ______________________________________________________________________
// Creating ir_maps
//
// This is the first pass and the one that drives the main
// computation.  It walks up and down the IR once.  On the way down,
// we count for each function the number of variables as well as
// liveness nodes.  A liveness node is basically an expression or
// capture clause that does something of interest: either it has
// interesting control flow or it uses/defines a local variable.
//
// On the way back up, at each function node we create liveness sets
// (we now know precisely how big to make our various vectors and so
// forth) and then do the data-flow propagation to compute the set
// of live variables at each program point.
//
// Finally, we run back over the IR one last time and, using the
// computed liveness, check various safety conditions.  For example,
// there must be no live nodes at the definition site for a variable
// unless it has an initializer.  Similarly, each non-mutable local
// variable must not be assigned if there is some successor
// assignment.  And so forth.

impl LiveNode {
    fn is_valid(&self) -> bool {
        self.get() != usize::MAX
    }
}

fn invalid_node() -> LiveNode { LiveNode(usize::MAX) }

struct CaptureInfo {
    ln: LiveNode,
    var_nid: NodeId
}

#[derive(Copy, Clone, Debug)]
struct LocalInfo {
    id: NodeId,
    name: ast::Name
}

#[derive(Copy, Clone, Debug)]
enum VarKind {
    Arg(NodeId, ast::Name),
    Local(LocalInfo),
    ImplicitRet,
    CleanExit
}

struct IrMaps<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,

    num_live_nodes: usize,
    num_vars: usize,
    live_node_map: NodeMap<LiveNode>,
    variable_map: NodeMap<Variable>,
    capture_info_map: NodeMap<Rc<Vec<CaptureInfo>>>,
    var_kinds: Vec<VarKind>,
    lnks: Vec<LiveNodeKind>,
}

impl<'a, 'tcx> IrMaps<'a, 'tcx> {
    fn new(tcx: &'a ty::ctxt<'tcx>) -> IrMaps<'a, 'tcx> {
        IrMaps {
            tcx: tcx,
            num_live_nodes: 0,
            num_vars: 0,
            live_node_map: NodeMap(),
            variable_map: NodeMap(),
            capture_info_map: NodeMap(),
            var_kinds: Vec::new(),
            lnks: Vec::new(),
        }
    }

    fn add_live_node(&mut self, lnk: LiveNodeKind) -> LiveNode {
        let ln = LiveNode(self.num_live_nodes);
        self.lnks.push(lnk);
        self.num_live_nodes += 1;

        debug!("{:?} is of kind {}", ln,
               live_node_kind_to_string(lnk, self.tcx));

        ln
    }

    fn add_live_node_for_node(&mut self, node_id: NodeId, lnk: LiveNodeKind) {
        let ln = self.add_live_node(lnk);
        self.live_node_map.insert(node_id, ln);

        debug!("{:?} is node {}", ln, node_id);
    }

    fn add_variable(&mut self, vk: VarKind) -> Variable {
        let v = Variable(self.num_vars);
        self.var_kinds.push(vk);
        self.num_vars += 1;

        match vk {
            Local(LocalInfo { id: node_id, .. }) | Arg(node_id, _) => {
                self.variable_map.insert(node_id, v);
            },
            ImplicitRet | CleanExit => {}
        }

        debug!("{:?} is {:?}", v, vk);

        v
    }

    fn variable(&self, node_id: NodeId, span: Span) -> Variable {
        match self.variable_map.get(&node_id) {
          Some(&var) => var,
          None => {
            self.tcx
                .sess
                .span_bug(span, &format!("no variable registered for id {}",
                                        node_id));
          }
        }
    }

    fn variable_name(&self, var: Variable) -> String {
        match self.var_kinds[var.get()] {
            Local(LocalInfo { name, .. }) | Arg(_, name) => {
                name.to_string()
            },
            ImplicitRet => "<implicit-ret>".to_string(),
            CleanExit => "<clean-exit>".to_string()
        }
    }

    fn set_captures(&mut self, node_id: NodeId, cs: Vec<CaptureInfo>) {
        self.capture_info_map.insert(node_id, Rc::new(cs));
    }

    fn lnk(&self, ln: LiveNode) -> LiveNodeKind {
        self.lnks[ln.get()]
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for Liveness<'a, 'tcx> {
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v hir::FnDecl,
                b: &'v hir::Block, s: Span, n: NodeId) {
        check_fn(self, fk, fd, b, s, n);
    }
    fn visit_local(&mut self, l: &hir::Local) {
        check_local(self, l);
    }
    fn visit_expr(&mut self, ex: &Expr) {
        check_expr(self, ex);
    }
    fn visit_arm(&mut self, a: &hir::Arm) {
        check_arm(self, a);
    }
}

fn visit_fn(ir: &mut IrMaps,
            fk: FnKind,
            decl: &hir::FnDecl,
            body: &hir::Block,
            sp: Span,
            id: ast::NodeId) {
    debug!("visit_fn");

    // swap in a new set of IR maps for this function body:
    let mut fn_maps = IrMaps::new(ir.tcx);

    debug!("creating fn_maps: {:?}", &fn_maps as *const IrMaps);

    for arg in &decl.inputs {
        pat_util::pat_bindings(&ir.tcx.def_map,
                               &*arg.pat,
                               |_bm, arg_id, _x, path1| {
            debug!("adding argument {}", arg_id);
            let name = path1.node;
            fn_maps.add_variable(Arg(arg_id, name));
        })
    };

    // gather up the various local variables, significant expressions,
    // and so forth:
    intravisit::walk_fn(&mut fn_maps, fk, decl, body, sp);

    // Special nodes and variables:
    // - exit_ln represents the end of the fn, either by return or panic
    // - implicit_ret_var is a pseudo-variable that represents
    //   an implicit return
    let specials = Specials {
        exit_ln: fn_maps.add_live_node(ExitNode),
        fallthrough_ln: fn_maps.add_live_node(ExitNode),
        no_ret_var: fn_maps.add_variable(ImplicitRet),
        clean_exit_var: fn_maps.add_variable(CleanExit)
    };

    // compute liveness
    let mut lsets = Liveness::new(&mut fn_maps, specials);
    let entry_ln = lsets.compute(decl, body);

    // check for various error conditions
    lsets.visit_block(body);
    lsets.check_ret(id, sp, fk, entry_ln, body);
    lsets.warn_about_unused_args(decl, entry_ln);
}

fn visit_local(ir: &mut IrMaps, local: &hir::Local) {
    pat_util::pat_bindings(&ir.tcx.def_map, &*local.pat, |_, p_id, sp, path1| {
        debug!("adding local variable {}", p_id);
        let name = path1.node;
        ir.add_live_node_for_node(p_id, VarDefNode(sp));
        ir.add_variable(Local(LocalInfo {
          id: p_id,
          name: name
        }));
    });
    intravisit::walk_local(ir, local);
}

fn visit_arm(ir: &mut IrMaps, arm: &hir::Arm) {
    for pat in &arm.pats {
        pat_util::pat_bindings(&ir.tcx.def_map, &**pat, |bm, p_id, sp, path1| {
            debug!("adding local variable {} from match with bm {:?}",
                   p_id, bm);
            let name = path1.node;
            ir.add_live_node_for_node(p_id, VarDefNode(sp));
            ir.add_variable(Local(LocalInfo {
                id: p_id,
                name: name
            }));
        })
    }
    intravisit::walk_arm(ir, arm);
}

fn visit_expr(ir: &mut IrMaps, expr: &Expr) {
    match expr.node {
      // live nodes required for uses or definitions of variables:
      hir::ExprPath(..) => {
        let def = ir.tcx.def_map.borrow().get(&expr.id).unwrap().full_def();
        debug!("expr {}: path that leads to {:?}", expr.id, def);
        if let DefLocal(..) = def {
            ir.add_live_node_for_node(expr.id, ExprNode(expr.span));
        }
        intravisit::walk_expr(ir, expr);
      }
      hir::ExprClosure(..) => {
        // Interesting control flow (for loops can contain labeled
        // breaks or continues)
        ir.add_live_node_for_node(expr.id, ExprNode(expr.span));

        // Make a live_node for each captured variable, with the span
        // being the location that the variable is used.  This results
        // in better error messages than just pointing at the closure
        // construction site.
        let mut call_caps = Vec::new();
        ir.tcx.with_freevars(expr.id, |freevars| {
            for fv in freevars {
                if let DefLocal(_, rv) = fv.def {
                    let fv_ln = ir.add_live_node(FreeVarNode(fv.span));
                    call_caps.push(CaptureInfo {ln: fv_ln,
                                                var_nid: rv});
                }
            }
        });
        ir.set_captures(expr.id, call_caps);

        intravisit::walk_expr(ir, expr);
      }

      // live nodes required for interesting control flow:
      hir::ExprIf(..) | hir::ExprMatch(..) | hir::ExprWhile(..) | hir::ExprLoop(..) => {
        ir.add_live_node_for_node(expr.id, ExprNode(expr.span));
        intravisit::walk_expr(ir, expr);
      }
      hir::ExprBinary(op, _, _) if ::rustc_front::util::lazy_binop(op.node) => {
        ir.add_live_node_for_node(expr.id, ExprNode(expr.span));
        intravisit::walk_expr(ir, expr);
      }

      // otherwise, live nodes are not required:
      hir::ExprIndex(..) | hir::ExprField(..) | hir::ExprTupField(..) |
      hir::ExprVec(..) | hir::ExprCall(..) | hir::ExprMethodCall(..) |
      hir::ExprTup(..) | hir::ExprBinary(..) | hir::ExprAddrOf(..) |
      hir::ExprCast(..) | hir::ExprUnary(..) | hir::ExprBreak(_) |
      hir::ExprAgain(_) | hir::ExprLit(_) | hir::ExprRet(..) |
      hir::ExprBlock(..) | hir::ExprAssign(..) | hir::ExprAssignOp(..) |
      hir::ExprStruct(..) | hir::ExprRepeat(..) |
      hir::ExprInlineAsm(..) | hir::ExprBox(..) |
      hir::ExprRange(..) => {
          intravisit::walk_expr(ir, expr);
      }
    }
}

// ______________________________________________________________________
// Computing liveness sets
//
// Actually we compute just a bit more than just liveness, but we use
// the same basic propagation framework in all cases.

#[derive(Clone, Copy)]
struct Users {
    reader: LiveNode,
    writer: LiveNode,
    used: bool
}

fn invalid_users() -> Users {
    Users {
        reader: invalid_node(),
        writer: invalid_node(),
        used: false
    }
}

#[derive(Copy, Clone)]
struct Specials {
    exit_ln: LiveNode,
    fallthrough_ln: LiveNode,
    no_ret_var: Variable,
    clean_exit_var: Variable
}

const ACC_READ: u32 = 1;
const ACC_WRITE: u32 = 2;
const ACC_USE: u32 = 4;

struct Liveness<'a, 'tcx: 'a> {
    ir: &'a mut IrMaps<'a, 'tcx>,
    s: Specials,
    successors: Vec<LiveNode>,
    users: Vec<Users>,
    // The list of node IDs for the nested loop scopes
    // we're in.
    loop_scope: Vec<NodeId>,
    // mappings from loop node ID to LiveNode
    // ("break" label should map to loop node ID,
    // it probably doesn't now)
    break_ln: NodeMap<LiveNode>,
    cont_ln: NodeMap<LiveNode>
}

impl<'a, 'tcx> Liveness<'a, 'tcx> {
    fn new(ir: &'a mut IrMaps<'a, 'tcx>, specials: Specials) -> Liveness<'a, 'tcx> {
        let num_live_nodes = ir.num_live_nodes;
        let num_vars = ir.num_vars;
        Liveness {
            ir: ir,
            s: specials,
            successors: vec![invalid_node(); num_live_nodes],
            users: vec![invalid_users(); num_live_nodes * num_vars],
            loop_scope: Vec::new(),
            break_ln: NodeMap(),
            cont_ln: NodeMap(),
        }
    }

    fn live_node(&self, node_id: NodeId, span: Span) -> LiveNode {
        match self.ir.live_node_map.get(&node_id) {
          Some(&ln) => ln,
          None => {
            // This must be a mismatch between the ir_map construction
            // above and the propagation code below; the two sets of
            // code have to agree about which AST nodes are worth
            // creating liveness nodes for.
            self.ir.tcx.sess.span_bug(
                span,
                &format!("no live node registered for node {}",
                        node_id));
          }
        }
    }

    fn variable(&self, node_id: NodeId, span: Span) -> Variable {
        self.ir.variable(node_id, span)
    }

    fn pat_bindings<F>(&mut self, pat: &hir::Pat, mut f: F) where
        F: FnMut(&mut Liveness<'a, 'tcx>, LiveNode, Variable, Span, NodeId),
    {
        pat_util::pat_bindings(&self.ir.tcx.def_map, pat, |_bm, p_id, sp, _n| {
            let ln = self.live_node(p_id, sp);
            let var = self.variable(p_id, sp);
            f(self, ln, var, sp, p_id);
        })
    }

    fn arm_pats_bindings<F>(&mut self, pat: Option<&hir::Pat>, f: F) where
        F: FnMut(&mut Liveness<'a, 'tcx>, LiveNode, Variable, Span, NodeId),
    {
        match pat {
            Some(pat) => {
                self.pat_bindings(pat, f);
            }
            None => {}
        }
    }

    fn define_bindings_in_pat(&mut self, pat: &hir::Pat, succ: LiveNode)
                              -> LiveNode {
        self.define_bindings_in_arm_pats(Some(pat), succ)
    }

    fn define_bindings_in_arm_pats(&mut self, pat: Option<&hir::Pat>, succ: LiveNode)
                                   -> LiveNode {
        let mut succ = succ;
        self.arm_pats_bindings(pat, |this, ln, var, _sp, _id| {
            this.init_from_succ(ln, succ);
            this.define(ln, var);
            succ = ln;
        });
        succ
    }

    fn idx(&self, ln: LiveNode, var: Variable) -> usize {
        ln.get() * self.ir.num_vars + var.get()
    }

    fn live_on_entry(&self, ln: LiveNode, var: Variable)
                      -> Option<LiveNodeKind> {
        assert!(ln.is_valid());
        let reader = self.users[self.idx(ln, var)].reader;
        if reader.is_valid() {Some(self.ir.lnk(reader))} else {None}
    }

    /*
    Is this variable live on entry to any of its successor nodes?
    */
    fn live_on_exit(&self, ln: LiveNode, var: Variable)
                    -> Option<LiveNodeKind> {
        let successor = self.successors[ln.get()];
        self.live_on_entry(successor, var)
    }

    fn used_on_entry(&self, ln: LiveNode, var: Variable) -> bool {
        assert!(ln.is_valid());
        self.users[self.idx(ln, var)].used
    }

    fn assigned_on_entry(&self, ln: LiveNode, var: Variable)
                         -> Option<LiveNodeKind> {
        assert!(ln.is_valid());
        let writer = self.users[self.idx(ln, var)].writer;
        if writer.is_valid() {Some(self.ir.lnk(writer))} else {None}
    }

    fn assigned_on_exit(&self, ln: LiveNode, var: Variable)
                        -> Option<LiveNodeKind> {
        let successor = self.successors[ln.get()];
        self.assigned_on_entry(successor, var)
    }

    fn indices2<F>(&mut self, ln: LiveNode, succ_ln: LiveNode, mut op: F) where
        F: FnMut(&mut Liveness<'a, 'tcx>, usize, usize),
    {
        let node_base_idx = self.idx(ln, Variable(0));
        let succ_base_idx = self.idx(succ_ln, Variable(0));
        for var_idx in 0..self.ir.num_vars {
            op(self, node_base_idx + var_idx, succ_base_idx + var_idx);
        }
    }

    fn write_vars<F>(&self,
                     wr: &mut Write,
                     ln: LiveNode,
                     mut test: F)
                     -> io::Result<()> where
        F: FnMut(usize) -> LiveNode,
    {
        let node_base_idx = self.idx(ln, Variable(0));
        for var_idx in 0..self.ir.num_vars {
            let idx = node_base_idx + var_idx;
            if test(idx).is_valid() {
                try!(write!(wr, " {:?}", Variable(var_idx)));
            }
        }
        Ok(())
    }

    fn find_loop_scope(&self,
                       opt_label: Option<ast::Name>,
                       id: NodeId,
                       sp: Span)
                       -> NodeId {
        match opt_label {
            Some(_) => {
                // Refers to a labeled loop. Use the results of resolve
                // to find with one
                match self.ir.tcx.def_map.borrow().get(&id).map(|d| d.full_def()) {
                    Some(DefLabel(loop_id)) => loop_id,
                    _ => self.ir.tcx.sess.span_bug(sp, "label on break/loop \
                                                        doesn't refer to a loop")
                }
            }
            None => {
                // Vanilla 'break' or 'loop', so use the enclosing
                // loop scope
                if self.loop_scope.is_empty() {
                    self.ir.tcx.sess.span_bug(sp, "break outside loop");
                } else {
                    *self.loop_scope.last().unwrap()
                }
            }
        }
    }

    #[allow(unused_must_use)]
    fn ln_str(&self, ln: LiveNode) -> String {
        let mut wr = Vec::new();
        {
            let wr = &mut wr as &mut Write;
            write!(wr, "[ln({:?}) of kind {:?} reads", ln.get(), self.ir.lnk(ln));
            self.write_vars(wr, ln, |idx| self.users[idx].reader);
            write!(wr, "  writes");
            self.write_vars(wr, ln, |idx| self.users[idx].writer);
            write!(wr, "  precedes {:?}]", self.successors[ln.get()]);
        }
        String::from_utf8(wr).unwrap()
    }

    fn init_empty(&mut self, ln: LiveNode, succ_ln: LiveNode) {
        self.successors[ln.get()] = succ_ln;

        // It is not necessary to initialize the
        // values to empty because this is the value
        // they have when they are created, and the sets
        // only grow during iterations.
        //
        // self.indices(ln) { |idx|
        //     self.users[idx] = invalid_users();
        // }
    }

    fn init_from_succ(&mut self, ln: LiveNode, succ_ln: LiveNode) {
        // more efficient version of init_empty() / merge_from_succ()
        self.successors[ln.get()] = succ_ln;

        self.indices2(ln, succ_ln, |this, idx, succ_idx| {
            this.users[idx] = this.users[succ_idx]
        });
        debug!("init_from_succ(ln={}, succ={})",
               self.ln_str(ln), self.ln_str(succ_ln));
    }

    fn merge_from_succ(&mut self,
                       ln: LiveNode,
                       succ_ln: LiveNode,
                       first_merge: bool)
                       -> bool {
        if ln == succ_ln { return false; }

        let mut changed = false;
        self.indices2(ln, succ_ln, |this, idx, succ_idx| {
            changed |= copy_if_invalid(this.users[succ_idx].reader,
                                       &mut this.users[idx].reader);
            changed |= copy_if_invalid(this.users[succ_idx].writer,
                                       &mut this.users[idx].writer);
            if this.users[succ_idx].used && !this.users[idx].used {
                this.users[idx].used = true;
                changed = true;
            }
        });

        debug!("merge_from_succ(ln={:?}, succ={}, first_merge={}, changed={})",
               ln, self.ln_str(succ_ln), first_merge, changed);
        return changed;

        fn copy_if_invalid(src: LiveNode, dst: &mut LiveNode) -> bool {
            if src.is_valid() && !dst.is_valid() {
                *dst = src;
                true
            } else {
                false
            }
        }
    }

    // Indicates that a local variable was *defined*; we know that no
    // uses of the variable can precede the definition (resolve checks
    // this) so we just clear out all the data.
    fn define(&mut self, writer: LiveNode, var: Variable) {
        let idx = self.idx(writer, var);
        self.users[idx].reader = invalid_node();
        self.users[idx].writer = invalid_node();

        debug!("{:?} defines {:?} (idx={}): {}", writer, var,
               idx, self.ln_str(writer));
    }

    // Either read, write, or both depending on the acc bitset
    fn acc(&mut self, ln: LiveNode, var: Variable, acc: u32) {
        debug!("{:?} accesses[{:x}] {:?}: {}",
               ln, acc, var, self.ln_str(ln));

        let idx = self.idx(ln, var);
        let user = &mut self.users[idx];

        if (acc & ACC_WRITE) != 0 {
            user.reader = invalid_node();
            user.writer = ln;
        }

        // Important: if we both read/write, must do read second
        // or else the write will override.
        if (acc & ACC_READ) != 0 {
            user.reader = ln;
        }

        if (acc & ACC_USE) != 0 {
            user.used = true;
        }
    }

    // _______________________________________________________________________

    fn compute(&mut self, decl: &hir::FnDecl, body: &hir::Block) -> LiveNode {
        // if there is a `break` or `again` at the top level, then it's
        // effectively a return---this only occurs in `for` loops,
        // where the body is really a closure.

        debug!("compute: using id for block, {}", block_to_string(body));

        let exit_ln = self.s.exit_ln;
        let entry_ln: LiveNode =
            self.with_loop_nodes(body.id, exit_ln, exit_ln,
              |this| this.propagate_through_fn_block(decl, body));

        // hack to skip the loop unless debug! is enabled:
        debug!("^^ liveness computation results for body {} (entry={:?})",
               {
                   for ln_idx in 0..self.ir.num_live_nodes {
                       debug!("{:?}", self.ln_str(LiveNode(ln_idx)));
                   }
                   body.id
               },
               entry_ln);

        entry_ln
    }

    fn propagate_through_fn_block(&mut self, _: &hir::FnDecl, blk: &hir::Block)
                                  -> LiveNode {
        // the fallthrough exit is only for those cases where we do not
        // explicitly return:
        let s = self.s;
        self.init_from_succ(s.fallthrough_ln, s.exit_ln);
        if blk.expr.is_none() {
            self.acc(s.fallthrough_ln, s.no_ret_var, ACC_READ)
        }
        self.acc(s.fallthrough_ln, s.clean_exit_var, ACC_READ);

        self.propagate_through_block(blk, s.fallthrough_ln)
    }

    fn propagate_through_block(&mut self, blk: &hir::Block, succ: LiveNode)
                               -> LiveNode {
        let succ = self.propagate_through_opt_expr(blk.expr.as_ref().map(|e| &**e), succ);
        blk.stmts.iter().rev().fold(succ, |succ, stmt| {
            self.propagate_through_stmt(&**stmt, succ)
        })
    }

    fn propagate_through_stmt(&mut self, stmt: &hir::Stmt, succ: LiveNode)
                              -> LiveNode {
        match stmt.node {
            hir::StmtDecl(ref decl, _) => {
                self.propagate_through_decl(&**decl, succ)
            }

            hir::StmtExpr(ref expr, _) | hir::StmtSemi(ref expr, _) => {
                self.propagate_through_expr(&**expr, succ)
            }
        }
    }

    fn propagate_through_decl(&mut self, decl: &hir::Decl, succ: LiveNode)
                              -> LiveNode {
        match decl.node {
            hir::DeclLocal(ref local) => {
                self.propagate_through_local(&**local, succ)
            }
            hir::DeclItem(_) => succ,
        }
    }

    fn propagate_through_local(&mut self, local: &hir::Local, succ: LiveNode)
                               -> LiveNode {
        // Note: we mark the variable as defined regardless of whether
        // there is an initializer.  Initially I had thought to only mark
        // the live variable as defined if it was initialized, and then we
        // could check for uninit variables just by scanning what is live
        // at the start of the function. But that doesn't work so well for
        // immutable variables defined in a loop:
        //     loop { let x; x = 5; }
        // because the "assignment" loops back around and generates an error.
        //
        // So now we just check that variables defined w/o an
        // initializer are not live at the point of their
        // initialization, which is mildly more complex than checking
        // once at the func header but otherwise equivalent.

        let succ = self.propagate_through_opt_expr(local.init.as_ref().map(|e| &**e), succ);
        self.define_bindings_in_pat(&*local.pat, succ)
    }

    fn propagate_through_exprs(&mut self, exprs: &[P<Expr>], succ: LiveNode)
                               -> LiveNode {
        exprs.iter().rev().fold(succ, |succ, expr| {
            self.propagate_through_expr(&**expr, succ)
        })
    }

    fn propagate_through_opt_expr(&mut self,
                                  opt_expr: Option<&Expr>,
                                  succ: LiveNode)
                                  -> LiveNode {
        opt_expr.map_or(succ, |expr| self.propagate_through_expr(expr, succ))
    }

    fn propagate_through_expr(&mut self, expr: &Expr, succ: LiveNode)
                              -> LiveNode {
        debug!("propagate_through_expr: {}", expr_to_string(expr));

        match expr.node {
          // Interesting cases with control flow or which gen/kill

          hir::ExprPath(..) => {
              self.access_path(expr, succ, ACC_READ | ACC_USE)
          }

          hir::ExprField(ref e, _) => {
              self.propagate_through_expr(&**e, succ)
          }

          hir::ExprTupField(ref e, _) => {
              self.propagate_through_expr(&**e, succ)
          }

          hir::ExprClosure(_, _, ref blk) => {
              debug!("{} is an ExprClosure",
                     expr_to_string(expr));

              /*
              The next-node for a break is the successor of the entire
              loop. The next-node for a continue is the top of this loop.
              */
              let node = self.live_node(expr.id, expr.span);
              self.with_loop_nodes(blk.id, succ, node, |this| {

                 // the construction of a closure itself is not important,
                 // but we have to consider the closed over variables.
                 let caps = match this.ir.capture_info_map.get(&expr.id) {
                    Some(caps) => caps.clone(),
                    None => {
                        this.ir.tcx.sess.span_bug(expr.span, "no registered caps");
                     }
                 };
                 caps.iter().rev().fold(succ, |succ, cap| {
                     this.init_from_succ(cap.ln, succ);
                     let var = this.variable(cap.var_nid, expr.span);
                     this.acc(cap.ln, var, ACC_READ | ACC_USE);
                     cap.ln
                 })
              })
          }

          hir::ExprIf(ref cond, ref then, ref els) => {
            //
            //     (cond)
            //       |
            //       v
            //     (expr)
            //     /   \
            //    |     |
            //    v     v
            //  (then)(els)
            //    |     |
            //    v     v
            //   (  succ  )
            //
            let else_ln = self.propagate_through_opt_expr(els.as_ref().map(|e| &**e), succ);
            let then_ln = self.propagate_through_block(&**then, succ);
            let ln = self.live_node(expr.id, expr.span);
            self.init_from_succ(ln, else_ln);
            self.merge_from_succ(ln, then_ln, false);
            self.propagate_through_expr(&**cond, ln)
          }

          hir::ExprWhile(ref cond, ref blk, _) => {
            self.propagate_through_loop(expr, WhileLoop(&**cond), &**blk, succ)
          }

          // Note that labels have been resolved, so we don't need to look
          // at the label ident
          hir::ExprLoop(ref blk, _) => {
            self.propagate_through_loop(expr, LoopLoop, &**blk, succ)
          }

          hir::ExprMatch(ref e, ref arms, _) => {
            //
            //      (e)
            //       |
            //       v
            //     (expr)
            //     / | \
            //    |  |  |
            //    v  v  v
            //   (..arms..)
            //    |  |  |
            //    v  v  v
            //   (  succ  )
            //
            //
            let ln = self.live_node(expr.id, expr.span);
            self.init_empty(ln, succ);
            let mut first_merge = true;
            for arm in arms {
                let body_succ =
                    self.propagate_through_expr(&*arm.body, succ);
                let guard_succ =
                    self.propagate_through_opt_expr(arm.guard.as_ref().map(|e| &**e), body_succ);
                // only consider the first pattern; any later patterns must have
                // the same bindings, and we also consider the first pattern to be
                // the "authoritative" set of ids
                let arm_succ =
                    self.define_bindings_in_arm_pats(arm.pats.first().map(|p| &**p),
                                                     guard_succ);
                self.merge_from_succ(ln, arm_succ, first_merge);
                first_merge = false;
            };
            self.propagate_through_expr(&**e, ln)
          }

          hir::ExprRet(ref o_e) => {
            // ignore succ and subst exit_ln:
            let exit_ln = self.s.exit_ln;
            self.propagate_through_opt_expr(o_e.as_ref().map(|e| &**e), exit_ln)
          }

          hir::ExprBreak(opt_label) => {
              // Find which label this break jumps to
              let sc = self.find_loop_scope(opt_label.map(|l| l.node.name), expr.id, expr.span);

              // Now that we know the label we're going to,
              // look it up in the break loop nodes table

              match self.break_ln.get(&sc) {
                  Some(&b) => b,
                  None => self.ir.tcx.sess.span_bug(expr.span,
                                                    "break to unknown label")
              }
          }

          hir::ExprAgain(opt_label) => {
              // Find which label this expr continues to
              let sc = self.find_loop_scope(opt_label.map(|l| l.node.name), expr.id, expr.span);

              // Now that we know the label we're going to,
              // look it up in the continue loop nodes table

              match self.cont_ln.get(&sc) {
                  Some(&b) => b,
                  None => self.ir.tcx.sess.span_bug(expr.span,
                                                    "loop to unknown label")
              }
          }

          hir::ExprAssign(ref l, ref r) => {
            // see comment on lvalues in
            // propagate_through_lvalue_components()
            let succ = self.write_lvalue(&**l, succ, ACC_WRITE);
            let succ = self.propagate_through_lvalue_components(&**l, succ);
            self.propagate_through_expr(&**r, succ)
          }

          hir::ExprAssignOp(_, ref l, ref r) => {
            // see comment on lvalues in
            // propagate_through_lvalue_components()
            let succ = self.write_lvalue(&**l, succ, ACC_WRITE|ACC_READ);
            let succ = self.propagate_through_expr(&**r, succ);
            self.propagate_through_lvalue_components(&**l, succ)
          }

          // Uninteresting cases: just propagate in rev exec order

          hir::ExprVec(ref exprs) => {
            self.propagate_through_exprs(&exprs[..], succ)
          }

          hir::ExprRepeat(ref element, ref count) => {
            let succ = self.propagate_through_expr(&**count, succ);
            self.propagate_through_expr(&**element, succ)
          }

          hir::ExprStruct(_, ref fields, ref with_expr) => {
            let succ = self.propagate_through_opt_expr(with_expr.as_ref().map(|e| &**e), succ);
            fields.iter().rev().fold(succ, |succ, field| {
                self.propagate_through_expr(&*field.expr, succ)
            })
          }

          hir::ExprCall(ref f, ref args) => {
            let diverges = !self.ir.tcx.is_method_call(expr.id) &&
                self.ir.tcx.expr_ty_adjusted(&**f).fn_ret().diverges();
            let succ = if diverges {
                self.s.exit_ln
            } else {
                succ
            };
            let succ = self.propagate_through_exprs(&args[..], succ);
            self.propagate_through_expr(&**f, succ)
          }

          hir::ExprMethodCall(_, _, ref args) => {
            let method_call = ty::MethodCall::expr(expr.id);
            let method_ty = self.ir.tcx.tables.borrow().method_map[&method_call].ty;
            let succ = if method_ty.fn_ret().diverges() {
                self.s.exit_ln
            } else {
                succ
            };
            self.propagate_through_exprs(&args[..], succ)
          }

          hir::ExprTup(ref exprs) => {
            self.propagate_through_exprs(&exprs[..], succ)
          }

          hir::ExprBinary(op, ref l, ref r) if ::rustc_front::util::lazy_binop(op.node) => {
            let r_succ = self.propagate_through_expr(&**r, succ);

            let ln = self.live_node(expr.id, expr.span);
            self.init_from_succ(ln, succ);
            self.merge_from_succ(ln, r_succ, false);

            self.propagate_through_expr(&**l, ln)
          }

          hir::ExprIndex(ref l, ref r) |
          hir::ExprBinary(_, ref l, ref r) => {
            let r_succ = self.propagate_through_expr(&**r, succ);
            self.propagate_through_expr(&**l, r_succ)
          }

          hir::ExprRange(ref e1, ref e2) => {
            let succ = e2.as_ref().map_or(succ, |e| self.propagate_through_expr(&**e, succ));
            e1.as_ref().map_or(succ, |e| self.propagate_through_expr(&**e, succ))
          }

          hir::ExprBox(ref e) |
          hir::ExprAddrOf(_, ref e) |
          hir::ExprCast(ref e, _) |
          hir::ExprUnary(_, ref e) => {
            self.propagate_through_expr(&**e, succ)
          }

          hir::ExprInlineAsm(ref ia) => {

            let succ = ia.outputs.iter().rev().fold(succ, |succ, &(_, ref expr, _)| {
                // see comment on lvalues
                // in propagate_through_lvalue_components()
                let succ = self.write_lvalue(&**expr, succ, ACC_WRITE);
                self.propagate_through_lvalue_components(&**expr, succ)
            });
            // Inputs are executed first. Propagate last because of rev order
            ia.inputs.iter().rev().fold(succ, |succ, &(_, ref expr)| {
                self.propagate_through_expr(&**expr, succ)
            })
          }

          hir::ExprLit(..) => {
            succ
          }

          hir::ExprBlock(ref blk) => {
            self.propagate_through_block(&**blk, succ)
          }
        }
    }

    fn propagate_through_lvalue_components(&mut self,
                                           expr: &Expr,
                                           succ: LiveNode)
                                           -> LiveNode {
        // # Lvalues
        //
        // In general, the full flow graph structure for an
        // assignment/move/etc can be handled in one of two ways,
        // depending on whether what is being assigned is a "tracked
        // value" or not. A tracked value is basically a local
        // variable or argument.
        //
        // The two kinds of graphs are:
        //
        //    Tracked lvalue          Untracked lvalue
        // ----------------------++-----------------------
        //                       ||
        //         |             ||           |
        //         v             ||           v
        //     (rvalue)          ||       (rvalue)
        //         |             ||           |
        //         v             ||           v
        // (write of lvalue)     ||   (lvalue components)
        //         |             ||           |
        //         v             ||           v
        //      (succ)           ||        (succ)
        //                       ||
        // ----------------------++-----------------------
        //
        // I will cover the two cases in turn:
        //
        // # Tracked lvalues
        //
        // A tracked lvalue is a local variable/argument `x`.  In
        // these cases, the link_node where the write occurs is linked
        // to node id of `x`.  The `write_lvalue()` routine generates
        // the contents of this node.  There are no subcomponents to
        // consider.
        //
        // # Non-tracked lvalues
        //
        // These are lvalues like `x[5]` or `x.f`.  In that case, we
        // basically ignore the value which is written to but generate
        // reads for the components---`x` in these two examples.  The
        // components reads are generated by
        // `propagate_through_lvalue_components()` (this fn).
        //
        // # Illegal lvalues
        //
        // It is still possible to observe assignments to non-lvalues;
        // these errors are detected in the later pass borrowck.  We
        // just ignore such cases and treat them as reads.

        match expr.node {
            hir::ExprPath(..) => succ,
            hir::ExprField(ref e, _) => self.propagate_through_expr(&**e, succ),
            hir::ExprTupField(ref e, _) => self.propagate_through_expr(&**e, succ),
            _ => self.propagate_through_expr(expr, succ)
        }
    }

    // see comment on propagate_through_lvalue()
    fn write_lvalue(&mut self, expr: &Expr, succ: LiveNode, acc: u32)
                    -> LiveNode {
        match expr.node {
          hir::ExprPath(..) => {
              self.access_path(expr, succ, acc)
          }

          // We do not track other lvalues, so just propagate through
          // to their subcomponents.  Also, it may happen that
          // non-lvalues occur here, because those are detected in the
          // later pass borrowck.
          _ => succ
        }
    }

    fn access_path(&mut self, expr: &Expr, succ: LiveNode, acc: u32)
                   -> LiveNode {
        match self.ir.tcx.def_map.borrow().get(&expr.id).unwrap().full_def() {
          DefLocal(_, nid) => {
            let ln = self.live_node(expr.id, expr.span);
            if acc != 0 {
                self.init_from_succ(ln, succ);
                let var = self.variable(nid, expr.span);
                self.acc(ln, var, acc);
            }
            ln
          }
          _ => succ
        }
    }

    fn propagate_through_loop(&mut self,
                              expr: &Expr,
                              kind: LoopKind,
                              body: &hir::Block,
                              succ: LiveNode)
                              -> LiveNode {

        /*

        We model control flow like this:

              (cond) <--+
                |       |
                v       |
          +-- (expr)    |
          |     |       |
          |     v       |
          |   (body) ---+
          |
          |
          v
        (succ)

        */


        // first iteration:
        let mut first_merge = true;
        let ln = self.live_node(expr.id, expr.span);
        self.init_empty(ln, succ);
        match kind {
            LoopLoop => {}
            _ => {
                // If this is not a `loop` loop, then it's possible we bypass
                // the body altogether. Otherwise, the only way is via a `break`
                // in the loop body.
                self.merge_from_succ(ln, succ, first_merge);
                first_merge = false;
            }
        }
        debug!("propagate_through_loop: using id for loop body {} {}",
               expr.id, block_to_string(body));

        let cond_ln = match kind {
            LoopLoop => ln,
            WhileLoop(ref cond) => self.propagate_through_expr(&**cond, ln),
        };
        let body_ln = self.with_loop_nodes(expr.id, succ, ln, |this| {
            this.propagate_through_block(body, cond_ln)
        });

        // repeat until fixed point is reached:
        while self.merge_from_succ(ln, body_ln, first_merge) {
            first_merge = false;

            let new_cond_ln = match kind {
                LoopLoop => ln,
                WhileLoop(ref cond) => {
                    self.propagate_through_expr(&**cond, ln)
                }
            };
            assert!(cond_ln == new_cond_ln);
            assert!(body_ln == self.with_loop_nodes(expr.id, succ, ln,
            |this| this.propagate_through_block(body, cond_ln)));
        }

        cond_ln
    }

    fn with_loop_nodes<R, F>(&mut self,
                             loop_node_id: NodeId,
                             break_ln: LiveNode,
                             cont_ln: LiveNode,
                             f: F)
                             -> R where
        F: FnOnce(&mut Liveness<'a, 'tcx>) -> R,
    {
        debug!("with_loop_nodes: {} {}", loop_node_id, break_ln.get());
        self.loop_scope.push(loop_node_id);
        self.break_ln.insert(loop_node_id, break_ln);
        self.cont_ln.insert(loop_node_id, cont_ln);
        let r = f(self);
        self.loop_scope.pop();
        r
    }
}

// _______________________________________________________________________
// Checking for error conditions

fn check_local(this: &mut Liveness, local: &hir::Local) {
    match local.init {
        Some(_) => {
            this.warn_about_unused_or_dead_vars_in_pat(&*local.pat);
        },
        None => {
            this.pat_bindings(&*local.pat, |this, ln, var, sp, id| {
                this.warn_about_unused(sp, id, ln, var);
            })
        }
    }

    intravisit::walk_local(this, local);
}

fn check_arm(this: &mut Liveness, arm: &hir::Arm) {
    // only consider the first pattern; any later patterns must have
    // the same bindings, and we also consider the first pattern to be
    // the "authoritative" set of ids
    this.arm_pats_bindings(arm.pats.first().map(|p| &**p), |this, ln, var, sp, id| {
        this.warn_about_unused(sp, id, ln, var);
    });
    intravisit::walk_arm(this, arm);
}

fn check_expr(this: &mut Liveness, expr: &Expr) {
    match expr.node {
      hir::ExprAssign(ref l, _) => {
        this.check_lvalue(&**l);

        intravisit::walk_expr(this, expr);
      }

      hir::ExprAssignOp(_, ref l, _) => {
        this.check_lvalue(&**l);

        intravisit::walk_expr(this, expr);
      }

      hir::ExprInlineAsm(ref ia) => {
        for &(_, ref input) in &ia.inputs {
          this.visit_expr(&**input);
        }

        // Output operands must be lvalues
        for &(_, ref out, _) in &ia.outputs {
          this.check_lvalue(&**out);
          this.visit_expr(&**out);
        }

        intravisit::walk_expr(this, expr);
      }

      // no correctness conditions related to liveness
      hir::ExprCall(..) | hir::ExprMethodCall(..) | hir::ExprIf(..) |
      hir::ExprMatch(..) | hir::ExprWhile(..) | hir::ExprLoop(..) |
      hir::ExprIndex(..) | hir::ExprField(..) | hir::ExprTupField(..) |
      hir::ExprVec(..) | hir::ExprTup(..) | hir::ExprBinary(..) |
      hir::ExprCast(..) | hir::ExprUnary(..) | hir::ExprRet(..) |
      hir::ExprBreak(..) | hir::ExprAgain(..) | hir::ExprLit(_) |
      hir::ExprBlock(..) | hir::ExprAddrOf(..) |
      hir::ExprStruct(..) | hir::ExprRepeat(..) |
      hir::ExprClosure(..) | hir::ExprPath(..) | hir::ExprBox(..) |
      hir::ExprRange(..) => {
        intravisit::walk_expr(this, expr);
      }
    }
}

fn check_fn(_v: &Liveness,
            _fk: FnKind,
            _decl: &hir::FnDecl,
            _body: &hir::Block,
            _sp: Span,
            _id: NodeId) {
    // do not check contents of nested fns
}

impl<'a, 'tcx> Liveness<'a, 'tcx> {
    fn fn_ret(&self, id: NodeId) -> ty::PolyFnOutput<'tcx> {
        let fn_ty = self.ir.tcx.node_id_to_type(id);
        match fn_ty.sty {
            ty::TyClosure(closure_def_id, ref substs) =>
                self.ir.tcx.closure_type(closure_def_id, substs).sig.output(),
            _ => fn_ty.fn_ret()
        }
    }

    fn check_ret(&self,
                 id: NodeId,
                 sp: Span,
                 _fk: FnKind,
                 entry_ln: LiveNode,
                 body: &hir::Block)
    {
        // within the fn body, late-bound regions are liberated:
        let fn_ret =
            self.ir.tcx.liberate_late_bound_regions(
                self.ir.tcx.region_maps.item_extent(body.id),
                &self.fn_ret(id));

        match fn_ret {
            ty::FnConverging(t_ret)
                if self.live_on_entry(entry_ln, self.s.no_ret_var).is_some() => {

                if t_ret.is_nil() {
                    // for nil return types, it is ok to not return a value expl.
                } else {
                    let ends_with_stmt = match body.expr {
                        None if !body.stmts.is_empty() =>
                            match body.stmts.first().unwrap().node {
                                hir::StmtSemi(ref e, _) => {
                                    self.ir.tcx.expr_ty(&**e) == t_ret
                                },
                                _ => false
                            },
                        _ => false
                    };
                    span_err!(self.ir.tcx.sess, sp, E0269, "not all control paths return a value");
                    if ends_with_stmt {
                        let last_stmt = body.stmts.first().unwrap();
                        let original_span = original_sp(self.ir.tcx.sess.codemap(),
                                                        last_stmt.span, sp);
                        let span_semicolon = Span {
                            lo: original_span.hi - BytePos(1),
                            hi: original_span.hi,
                            expn_id: original_span.expn_id
                        };
                        self.ir.tcx.sess.span_help(
                            span_semicolon, "consider removing this semicolon:");
                    }
                }
            }
            ty::FnDiverging
                if self.live_on_entry(entry_ln, self.s.clean_exit_var).is_some() => {
                    span_err!(self.ir.tcx.sess, sp, E0270,
                        "computation may converge in a function marked as diverging");
                }

            _ => {}
        }
    }

    fn check_lvalue(&mut self, expr: &Expr) {
        match expr.node {
            hir::ExprPath(..) => {
                if let DefLocal(_, nid) = self.ir.tcx.def_map.borrow().get(&expr.id)
                                                                      .unwrap()
                                                                      .full_def() {
                    // Assignment to an immutable variable or argument: only legal
                    // if there is no later assignment. If this local is actually
                    // mutable, then check for a reassignment to flag the mutability
                    // as being used.
                    let ln = self.live_node(expr.id, expr.span);
                    let var = self.variable(nid, expr.span);
                    self.warn_about_dead_assign(expr.span, expr.id, ln, var);
                }
            }
            _ => {
                // For other kinds of lvalues, no checks are required,
                // and any embedded expressions are actually rvalues
                intravisit::walk_expr(self, expr);
            }
        }
    }

    fn should_warn(&self, var: Variable) -> Option<String> {
        let name = self.ir.variable_name(var);
        if name.is_empty() || name.as_bytes()[0] == ('_' as u8) {
            None
        } else {
            Some(name)
        }
    }

    fn warn_about_unused_args(&self, decl: &hir::FnDecl, entry_ln: LiveNode) {
        for arg in &decl.inputs {
            pat_util::pat_bindings(&self.ir.tcx.def_map,
                                   &*arg.pat,
                                   |_bm, p_id, sp, path1| {
                let var = self.variable(p_id, sp);
                // Ignore unused self.
                let name = path1.node;
                if name != special_idents::self_.name {
                    if !self.warn_about_unused(sp, p_id, entry_ln, var) {
                        if self.live_on_entry(entry_ln, var).is_none() {
                            self.report_dead_assign(p_id, sp, var, true);
                        }
                    }
                }
            })
        }
    }

    fn warn_about_unused_or_dead_vars_in_pat(&mut self, pat: &hir::Pat) {
        self.pat_bindings(pat, |this, ln, var, sp, id| {
            if !this.warn_about_unused(sp, id, ln, var) {
                this.warn_about_dead_assign(sp, id, ln, var);
            }
        })
    }

    fn warn_about_unused(&self,
                         sp: Span,
                         id: NodeId,
                         ln: LiveNode,
                         var: Variable)
                         -> bool {
        if !self.used_on_entry(ln, var) {
            let r = self.should_warn(var);
            if let Some(name) = r {

                // annoying: for parameters in funcs like `fn(x: int)
                // {ret}`, there is only one node, so asking about
                // assigned_on_exit() is not meaningful.
                let is_assigned = if ln == self.s.exit_ln {
                    false
                } else {
                    self.assigned_on_exit(ln, var).is_some()
                };

                if is_assigned {
                    self.ir.tcx.sess.add_lint(lint::builtin::UNUSED_VARIABLES, id, sp,
                        format!("variable `{}` is assigned to, but never used",
                                name));
                } else {
                    self.ir.tcx.sess.add_lint(lint::builtin::UNUSED_VARIABLES, id, sp,
                        format!("unused variable: `{}`", name));
                }
            }
            true
        } else {
            false
        }
    }

    fn warn_about_dead_assign(&self,
                              sp: Span,
                              id: NodeId,
                              ln: LiveNode,
                              var: Variable) {
        if self.live_on_exit(ln, var).is_none() {
            self.report_dead_assign(id, sp, var, false);
        }
    }

    fn report_dead_assign(&self, id: NodeId, sp: Span, var: Variable, is_argument: bool) {
        if let Some(name) = self.should_warn(var) {
            if is_argument {
                self.ir.tcx.sess.add_lint(lint::builtin::UNUSED_ASSIGNMENTS, id, sp,
                    format!("value passed to `{}` is never read", name));
            } else {
                self.ir.tcx.sess.add_lint(lint::builtin::UNUSED_ASSIGNMENTS, id, sp,
                    format!("value assigned to `{}` is never read", name));
            }
        }
    }
}
