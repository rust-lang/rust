// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * A classic liveness analysis based on dataflow over the AST.  Computes,
 * for each local variable in a function, whether that variable is live
 * at a given point.  Program execution points are identified by their
 * id.
 *
 * # Basic idea
 *
 * The basic model is that each local variable is assigned an index.  We
 * represent sets of local variables using a vector indexed by this
 * index.  The value in the vector is either 0, indicating the variable
 * is dead, or the id of an expression that uses the variable.
 *
 * We conceptually walk over the AST in reverse execution order.  If we
 * find a use of a variable, we add it to the set of live variables.  If
 * we find an assignment to a variable, we remove it from the set of live
 * variables.  When we have to merge two flows, we take the union of
 * those two flows---if the variable is live on both paths, we simply
 * pick one id.  In the event of loops, we continue doing this until a
 * fixed point is reached.
 *
 * ## Checking initialization
 *
 * At the function entry point, all variables must be dead.  If this is
 * not the case, we can report an error using the id found in the set of
 * live variables, which identifies a use of the variable which is not
 * dominated by an assignment.
 *
 * ## Checking moves
 *
 * After each explicit move, the variable must be dead.
 *
 * ## Computing last uses
 *
 * Any use of the variable where the variable is dead afterwards is a
 * last use.
 *
 * # Implementation details
 *
 * The actual implementation contains two (nested) walks over the AST.
 * The outer walk has the job of building up the ir_maps instance for the
 * enclosing function.  On the way down the tree, it identifies those AST
 * nodes and variable IDs that will be needed for the liveness analysis
 * and assigns them contiguous IDs.  The liveness id for an AST node is
 * called a `live_node` (it's a newtype'd uint) and the id for a variable
 * is called a `variable` (another newtype'd uint).
 *
 * On the way back up the tree, as we are about to exit from a function
 * declaration we allocate a `liveness` instance.  Now that we know
 * precisely how many nodes and variables we need, we can allocate all
 * the various arrays that we will need to precisely the right size.  We then
 * perform the actual propagation on the `liveness` instance.
 *
 * This propagation is encoded in the various `propagate_through_*()`
 * methods.  It effectively does a reverse walk of the AST; whenever we
 * reach a loop node, we iterate until a fixed point is reached.
 *
 * ## The `Users` struct
 *
 * At each live node `N`, we track three pieces of information for each
 * variable `V` (these are encapsulated in the `Users` struct):
 *
 * - `reader`: the `LiveNode` ID of some node which will read the value
 *    that `V` holds on entry to `N`.  Formally: a node `M` such
 *    that there exists a path `P` from `N` to `M` where `P` does not
 *    write `V`.  If the `reader` is `invalid_node()`, then the current
 *    value will never be read (the variable is dead, essentially).
 *
 * - `writer`: the `LiveNode` ID of some node which will write the
 *    variable `V` and which is reachable from `N`.  Formally: a node `M`
 *    such that there exists a path `P` from `N` to `M` and `M` writes
 *    `V`.  If the `writer` is `invalid_node()`, then there is no writer
 *    of `V` that follows `N`.
 *
 * - `used`: a boolean value indicating whether `V` is *used*.  We
 *   distinguish a *read* from a *use* in that a *use* is some read that
 *   is not just used to generate a new value.  For example, `x += 1` is
 *   a read but not a use.  This is used to generate better warnings.
 *
 * ## Special Variables
 *
 * We generate various special variables for various, well, special purposes.
 * These are described in the `specials` struct:
 *
 * - `exit_ln`: a live node that is generated to represent every 'exit' from
 *   the function, whether it be by explicit return, fail, or other means.
 *
 * - `fallthrough_ln`: a live node that represents a fallthrough
 *
 * - `no_ret_var`: a synthetic variable that is only 'read' from, the
 *   fallthrough node.  This allows us to detect functions where we fail
 *   to return explicitly.
 */


use middle::lint::{UnusedVariable, DeadAssignment};
use middle::pat_util;
use middle::ty;
use middle::typeck;
use middle::moves;

use std::cast::transmute;
use std::cell::{Cell, RefCell};
use std::hashmap::HashMap;
use std::io;
use std::str;
use std::to_str;
use std::uint;
use std::vec;
use syntax::ast::*;
use syntax::codemap::Span;
use syntax::parse::token::special_idents;
use syntax::parse::token;
use syntax::print::pprust::{expr_to_str, block_to_str};
use syntax::{visit, ast_util};
use syntax::visit::{Visitor, FnKind};

#[deriving(Eq)]
struct Variable(uint);
#[deriving(Eq)]
struct LiveNode(uint);

impl Variable {
    fn get(&self) -> uint { let Variable(v) = *self; v }
}

impl LiveNode {
    fn get(&self) -> uint { let LiveNode(v) = *self; v }
}

impl Clone for LiveNode {
    fn clone(&self) -> LiveNode {
        LiveNode(self.get())
    }
}

#[deriving(Eq)]
enum LiveNodeKind {
    FreeVarNode(Span),
    ExprNode(Span),
    VarDefNode(Span),
    ExitNode
}

fn live_node_kind_to_str(lnk: LiveNodeKind, cx: ty::ctxt) -> ~str {
    let cm = cx.sess.codemap;
    match lnk {
        FreeVarNode(s) => format!("Free var node [{}]", cm.span_to_str(s)),
        ExprNode(s)    => format!("Expr node [{}]", cm.span_to_str(s)),
        VarDefNode(s)  => format!("Var def node [{}]", cm.span_to_str(s)),
        ExitNode       => ~"Exit node"
    }
}

struct LivenessVisitor;

impl Visitor<@IrMaps> for LivenessVisitor {
    fn visit_fn(&mut self, fk: &FnKind, fd: &FnDecl, b: &Block, s: Span, n: NodeId, e: @IrMaps) {
        visit_fn(self, fk, fd, b, s, n, e);
    }
    fn visit_local(&mut self, l: &Local, e: @IrMaps) { visit_local(self, l, e); }
    fn visit_expr(&mut self, ex: &Expr, e: @IrMaps) { visit_expr(self, ex, e); }
    fn visit_arm(&mut self, a: &Arm, e: @IrMaps) { visit_arm(self, a, e); }
}

pub fn check_crate(tcx: ty::ctxt,
                   method_map: typeck::method_map,
                   capture_map: moves::CaptureMap,
                   krate: &Crate) {
    let mut visitor = LivenessVisitor;

    let initial_maps = @IrMaps(tcx, method_map, capture_map);
    visit::walk_crate(&mut visitor, krate, initial_maps);
    tcx.sess.abort_if_errors();
}

impl to_str::ToStr for LiveNode {
    fn to_str(&self) -> ~str { format!("ln({})", self.get()) }
}

impl to_str::ToStr for Variable {
    fn to_str(&self) -> ~str { format!("v({})", self.get()) }
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
    pub fn is_valid(&self) -> bool {
        self.get() != uint::MAX
    }
}

fn invalid_node() -> LiveNode { LiveNode(uint::MAX) }

struct CaptureInfo {
    ln: LiveNode,
    is_move: bool,
    var_nid: NodeId
}

enum LocalKind {
    FromMatch(BindingMode),
    FromLetWithInitializer,
    FromLetNoInitializer
}

struct LocalInfo {
    id: NodeId,
    ident: Ident,
    is_mutbl: bool,
    kind: LocalKind,
}

enum VarKind {
    Arg(NodeId, Ident),
    Local(LocalInfo),
    ImplicitRet
}

struct IrMaps {
    tcx: ty::ctxt,
    method_map: typeck::method_map,
    capture_map: moves::CaptureMap,

    num_live_nodes: Cell<uint>,
    num_vars: Cell<uint>,
    live_node_map: RefCell<HashMap<NodeId, LiveNode>>,
    variable_map: RefCell<HashMap<NodeId, Variable>>,
    capture_info_map: RefCell<HashMap<NodeId, @~[CaptureInfo]>>,
    var_kinds: RefCell<~[VarKind]>,
    lnks: RefCell<~[LiveNodeKind]>,
}

fn IrMaps(tcx: ty::ctxt,
          method_map: typeck::method_map,
          capture_map: moves::CaptureMap)
       -> IrMaps {
    IrMaps {
        tcx: tcx,
        method_map: method_map,
        capture_map: capture_map,
        num_live_nodes: Cell::new(0),
        num_vars: Cell::new(0),
        live_node_map: RefCell::new(HashMap::new()),
        variable_map: RefCell::new(HashMap::new()),
        capture_info_map: RefCell::new(HashMap::new()),
        var_kinds: RefCell::new(~[]),
        lnks: RefCell::new(~[]),
    }
}

impl IrMaps {
    pub fn add_live_node(&self, lnk: LiveNodeKind) -> LiveNode {
        let num_live_nodes = self.num_live_nodes.get();
        let ln = LiveNode(num_live_nodes);
        let mut lnks = self.lnks.borrow_mut();
        lnks.get().push(lnk);
        self.num_live_nodes.set(num_live_nodes + 1);

        debug!("{} is of kind {}", ln.to_str(),
               live_node_kind_to_str(lnk, self.tcx));

        ln
    }

    pub fn add_live_node_for_node(&self, node_id: NodeId, lnk: LiveNodeKind) {
        let ln = self.add_live_node(lnk);
        let mut live_node_map = self.live_node_map.borrow_mut();
        live_node_map.get().insert(node_id, ln);

        debug!("{} is node {}", ln.to_str(), node_id);
    }

    pub fn add_variable(&self, vk: VarKind) -> Variable {
        let v = Variable(self.num_vars.get());
        {
            let mut var_kinds = self.var_kinds.borrow_mut();
            var_kinds.get().push(vk);
        }
        self.num_vars.set(self.num_vars.get() + 1);

        match vk {
            Local(LocalInfo { id: node_id, .. }) | Arg(node_id, _) => {
                let mut variable_map = self.variable_map.borrow_mut();
                variable_map.get().insert(node_id, v);
            },
            ImplicitRet => {}
        }

        debug!("{} is {:?}", v.to_str(), vk);

        v
    }

    pub fn variable(&self, node_id: NodeId, span: Span) -> Variable {
        let variable_map = self.variable_map.borrow();
        match variable_map.get().find(&node_id) {
          Some(&var) => var,
          None => {
            self.tcx.sess.span_bug(
                span, format!("no variable registered for id {}", node_id));
          }
        }
    }

    pub fn variable_name(&self, var: Variable) -> ~str {
        let var_kinds = self.var_kinds.borrow();
        match var_kinds.get()[var.get()] {
            Local(LocalInfo { ident: nm, .. }) | Arg(_, nm) => {
                let string = token::get_ident(nm.name);
                string.get().to_str()
            },
            ImplicitRet => ~"<implicit-ret>"
        }
    }

    pub fn set_captures(&self, node_id: NodeId, cs: ~[CaptureInfo]) {
        let mut capture_info_map = self.capture_info_map.borrow_mut();
        capture_info_map.get().insert(node_id, @cs);
    }

    pub fn captures(&self, expr: &Expr) -> @~[CaptureInfo] {
        let capture_info_map = self.capture_info_map.borrow();
        match capture_info_map.get().find(&expr.id) {
          Some(&caps) => caps,
          None => {
            self.tcx.sess.span_bug(expr.span, "no registered caps");
          }
        }
    }

    pub fn lnk(&self, ln: LiveNode) -> LiveNodeKind {
        let lnks = self.lnks.borrow();
        lnks.get()[ln.get()]
    }
}

impl Visitor<()> for Liveness {
    fn visit_fn(&mut self, fk: &FnKind, fd: &FnDecl, b: &Block, s: Span, n: NodeId, _: ()) {
        check_fn(self, fk, fd, b, s, n);
    }
    fn visit_local(&mut self, l: &Local, _: ()) {
        check_local(self, l);
    }
    fn visit_expr(&mut self, ex: &Expr, _: ()) {
        check_expr(self, ex);
    }
    fn visit_arm(&mut self, a: &Arm, _: ()) {
        check_arm(self, a);
    }
}

fn visit_fn(v: &mut LivenessVisitor,
            fk: &FnKind,
            decl: &FnDecl,
            body: &Block,
            sp: Span,
            id: NodeId,
            this: @IrMaps) {
    debug!("visit_fn: id={}", id);
    let _i = ::util::common::indenter();

    // swap in a new set of IR maps for this function body:
    let fn_maps = @IrMaps(this.tcx, this.method_map, this.capture_map);

    unsafe {
        debug!("creating fn_maps: {}", transmute::<&IrMaps, *IrMaps>(fn_maps));
    }

    for arg in decl.inputs.iter() {
        pat_util::pat_bindings(this.tcx.def_map,
                               arg.pat,
                               |_bm, arg_id, _x, path| {
            debug!("adding argument {}", arg_id);
            let ident = ast_util::path_to_ident(path);
            fn_maps.add_variable(Arg(arg_id, ident));
        })
    };

    // gather up the various local variables, significant expressions,
    // and so forth:
    visit::walk_fn(v, fk, decl, body, sp, id, fn_maps);

    // Special nodes and variables:
    // - exit_ln represents the end of the fn, either by return or fail
    // - implicit_ret_var is a pseudo-variable that represents
    //   an implicit return
    let specials = Specials {
        exit_ln: fn_maps.add_live_node(ExitNode),
        fallthrough_ln: fn_maps.add_live_node(ExitNode),
        no_ret_var: fn_maps.add_variable(ImplicitRet)
    };

    // compute liveness
    let mut lsets = Liveness(fn_maps, specials);
    let entry_ln = lsets.compute(decl, body);

    // check for various error conditions
    lsets.visit_block(body, ());
    lsets.check_ret(id, sp, fk, entry_ln, body);
    lsets.warn_about_unused_args(decl, entry_ln);
}

fn visit_local(v: &mut LivenessVisitor, local: &Local, this: @IrMaps) {
    let def_map = this.tcx.def_map;
    pat_util::pat_bindings(def_map, local.pat, |bm, p_id, sp, path| {
        debug!("adding local variable {}", p_id);
        let name = ast_util::path_to_ident(path);
        this.add_live_node_for_node(p_id, VarDefNode(sp));
        let kind = match local.init {
          Some(_) => FromLetWithInitializer,
          None => FromLetNoInitializer
        };
        let mutbl = match bm {
            BindByValue(MutMutable) => true,
            _ => false
        };
        this.add_variable(Local(LocalInfo {
          id: p_id,
          ident: name,
          is_mutbl: mutbl,
          kind: kind
        }));
    });
    visit::walk_local(v, local, this);
}

fn visit_arm(v: &mut LivenessVisitor, arm: &Arm, this: @IrMaps) {
    let def_map = this.tcx.def_map;
    for pat in arm.pats.iter() {
        pat_util::pat_bindings(def_map, *pat, |bm, p_id, sp, path| {
            debug!("adding local variable {} from match with bm {:?}",
                   p_id, bm);
            let name = ast_util::path_to_ident(path);
            let mutbl = match bm {
                BindByValue(MutMutable) => true,
                _ => false
            };
            this.add_live_node_for_node(p_id, VarDefNode(sp));
            this.add_variable(Local(LocalInfo {
                id: p_id,
                ident: name,
                is_mutbl: mutbl,
                kind: FromMatch(bm)
            }));
        })
    }
    visit::walk_arm(v, arm, this);
}

fn visit_expr(v: &mut LivenessVisitor, expr: &Expr, this: @IrMaps) {
    match expr.node {
      // live nodes required for uses or definitions of variables:
      ExprPath(_) => {
        let def_map = this.tcx.def_map.borrow();
        let def = def_map.get().get_copy(&expr.id);
        debug!("expr {}: path that leads to {:?}", expr.id, def);
        if moves::moved_variable_node_id_from_def(def).is_some() {
            this.add_live_node_for_node(expr.id, ExprNode(expr.span));
        }
        visit::walk_expr(v, expr, this);
      }
      ExprFnBlock(..) | ExprProc(..) => {
        // Interesting control flow (for loops can contain labeled
        // breaks or continues)
        this.add_live_node_for_node(expr.id, ExprNode(expr.span));

        // Make a live_node for each captured variable, with the span
        // being the location that the variable is used.  This results
        // in better error messages than just pointing at the closure
        // construction site.
        let capture_map = this.capture_map.borrow();
        let cvs = capture_map.get().get(&expr.id);
        let mut call_caps = ~[];
        for cv in cvs.borrow().iter() {
            match moves::moved_variable_node_id_from_def(cv.def) {
              Some(rv) => {
                let cv_ln = this.add_live_node(FreeVarNode(cv.span));
                let is_move = match cv.mode {
                    // var must be dead afterwards
                    moves::CapMove => true,

                    // var can stil be used
                    moves::CapCopy | moves::CapRef => false
                };
                call_caps.push(CaptureInfo {ln: cv_ln,
                                            is_move: is_move,
                                            var_nid: rv});
              }
              None => {}
            }
        }
        this.set_captures(expr.id, call_caps);

        visit::walk_expr(v, expr, this);
      }

      // live nodes required for interesting control flow:
      ExprIf(..) | ExprMatch(..) | ExprWhile(..) | ExprLoop(..) => {
        this.add_live_node_for_node(expr.id, ExprNode(expr.span));
        visit::walk_expr(v, expr, this);
      }
      ExprForLoop(..) => fail!("non-desugared expr_for_loop"),
      ExprBinary(_, op, _, _) if ast_util::lazy_binop(op) => {
        this.add_live_node_for_node(expr.id, ExprNode(expr.span));
        visit::walk_expr(v, expr, this);
      }

      // otherwise, live nodes are not required:
      ExprIndex(..) | ExprField(..) | ExprVstore(..) | ExprVec(..) |
      ExprCall(..) | ExprMethodCall(..) | ExprTup(..) | ExprLogLevel |
      ExprBinary(..) | ExprAddrOf(..) |
      ExprCast(..) | ExprUnary(..) | ExprBreak(_) |
      ExprAgain(_) | ExprLit(_) | ExprRet(..) | ExprBlock(..) |
      ExprAssign(..) | ExprAssignOp(..) | ExprMac(..) |
      ExprStruct(..) | ExprRepeat(..) | ExprParen(..) |
      ExprInlineAsm(..) | ExprBox(..) => {
          visit::walk_expr(v, expr, this);
      }
    }
}

// ______________________________________________________________________
// Computing liveness sets
//
// Actually we compute just a bit more than just liveness, but we use
// the same basic propagation framework in all cases.

#[deriving(Clone)]
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

struct Specials {
    exit_ln: LiveNode,
    fallthrough_ln: LiveNode,
    no_ret_var: Variable
}

static ACC_READ: uint = 1u;
static ACC_WRITE: uint = 2u;
static ACC_USE: uint = 4u;

type LiveNodeMap = @RefCell<HashMap<NodeId, LiveNode>>;

pub struct Liveness {
    tcx: ty::ctxt,
    ir: @IrMaps,
    s: Specials,
    successors: @RefCell<~[LiveNode]>,
    users: @RefCell<~[Users]>,
    // The list of node IDs for the nested loop scopes
    // we're in.
    loop_scope: @RefCell<~[NodeId]>,
    // mappings from loop node ID to LiveNode
    // ("break" label should map to loop node ID,
    // it probably doesn't now)
    break_ln: LiveNodeMap,
    cont_ln: LiveNodeMap
}

fn Liveness(ir: @IrMaps, specials: Specials) -> Liveness {
    Liveness {
        ir: ir,
        tcx: ir.tcx,
        s: specials,
        successors: @RefCell::new(vec::from_elem(ir.num_live_nodes.get(),
                                                 invalid_node())),
        users: @RefCell::new(vec::from_elem(ir.num_live_nodes.get() *
                                            ir.num_vars.get(),
                                            invalid_users())),
        loop_scope: @RefCell::new(~[]),
        break_ln: @RefCell::new(HashMap::new()),
        cont_ln: @RefCell::new(HashMap::new()),
    }
}

impl Liveness {
    pub fn live_node(&self, node_id: NodeId, span: Span) -> LiveNode {
        let ir: &IrMaps = self.ir;
        let live_node_map = ir.live_node_map.borrow();
        match live_node_map.get().find(&node_id) {
          Some(&ln) => ln,
          None => {
            // This must be a mismatch between the ir_map construction
            // above and the propagation code below; the two sets of
            // code have to agree about which AST nodes are worth
            // creating liveness nodes for.
            self.tcx.sess.span_bug(
                span, format!("no live node registered for node {}",
                           node_id));
          }
        }
    }

    pub fn variable(&self, node_id: NodeId, span: Span) -> Variable {
        self.ir.variable(node_id, span)
    }

    pub fn pat_bindings(&self,
                        pat: @Pat,
                        f: |LiveNode, Variable, Span, NodeId|) {
        let def_map = self.tcx.def_map;
        pat_util::pat_bindings(def_map, pat, |_bm, p_id, sp, _n| {
            let ln = self.live_node(p_id, sp);
            let var = self.variable(p_id, sp);
            f(ln, var, sp, p_id);
        })
    }

    pub fn arm_pats_bindings(&self,
                             pats: &[@Pat],
                             f: |LiveNode, Variable, Span, NodeId|) {
        // only consider the first pattern; any later patterns must have
        // the same bindings, and we also consider the first pattern to be
        // the "authoratative" set of ids
        if !pats.is_empty() {
            self.pat_bindings(pats[0], f)
        }
    }

    pub fn define_bindings_in_pat(&self, pat: @Pat, succ: LiveNode)
                                  -> LiveNode {
        self.define_bindings_in_arm_pats([pat], succ)
    }

    pub fn define_bindings_in_arm_pats(&self, pats: &[@Pat], succ: LiveNode)
                                       -> LiveNode {
        let mut succ = succ;
        self.arm_pats_bindings(pats, |ln, var, _sp, _id| {
            self.init_from_succ(ln, succ);
            self.define(ln, var);
            succ = ln;
        });
        succ
    }

    pub fn idx(&self, ln: LiveNode, var: Variable) -> uint {
        ln.get() * self.ir.num_vars.get() + var.get()
    }

    pub fn live_on_entry(&self, ln: LiveNode, var: Variable)
                         -> Option<LiveNodeKind> {
        assert!(ln.is_valid());
        let users = self.users.borrow();
        let reader = users.get()[self.idx(ln, var)].reader;
        if reader.is_valid() {Some(self.ir.lnk(reader))} else {None}
    }

    /*
    Is this variable live on entry to any of its successor nodes?
    */
    pub fn live_on_exit(&self, ln: LiveNode, var: Variable)
                        -> Option<LiveNodeKind> {
        let successor = {
            let successors = self.successors.borrow();
            successors.get()[ln.get()]
        };
        self.live_on_entry(successor, var)
    }

    pub fn used_on_entry(&self, ln: LiveNode, var: Variable) -> bool {
        assert!(ln.is_valid());
        let users = self.users.borrow();
        users.get()[self.idx(ln, var)].used
    }

    pub fn assigned_on_entry(&self, ln: LiveNode, var: Variable)
                             -> Option<LiveNodeKind> {
        assert!(ln.is_valid());
        let users = self.users.borrow();
        let writer = users.get()[self.idx(ln, var)].writer;
        if writer.is_valid() {Some(self.ir.lnk(writer))} else {None}
    }

    pub fn assigned_on_exit(&self, ln: LiveNode, var: Variable)
                            -> Option<LiveNodeKind> {
        let successor = {
            let successors = self.successors.borrow();
            successors.get()[ln.get()]
        };
        self.assigned_on_entry(successor, var)
    }

    pub fn indices2(&self,
                    ln: LiveNode,
                    succ_ln: LiveNode,
                    op: |uint, uint|) {
        let node_base_idx = self.idx(ln, Variable(0u));
        let succ_base_idx = self.idx(succ_ln, Variable(0u));
        for var_idx in range(0u, self.ir.num_vars.get()) {
            op(node_base_idx + var_idx, succ_base_idx + var_idx);
        }
    }

    pub fn write_vars(&self,
                      wr: &mut io::Writer,
                      ln: LiveNode,
                      test: |uint| -> LiveNode) -> io::IoResult<()> {
        let node_base_idx = self.idx(ln, Variable(0));
        for var_idx in range(0u, self.ir.num_vars.get()) {
            let idx = node_base_idx + var_idx;
            if test(idx).is_valid() {
                if_ok!(write!(wr, " {}", Variable(var_idx).to_str()));
            }
        }
        Ok(())
    }

    pub fn find_loop_scope(&self,
                           opt_label: Option<Name>,
                           id: NodeId,
                           sp: Span)
                           -> NodeId {
        match opt_label {
            Some(_) => {
                // Refers to a labeled loop. Use the results of resolve
                // to find with one
                let def_map = self.tcx.def_map.borrow();
                match def_map.get().find(&id) {
                    Some(&DefLabel(loop_id)) => loop_id,
                    _ => self.tcx.sess.span_bug(sp, "label on break/loop \
                                                     doesn't refer to a loop")
                }
            }
            None => {
                // Vanilla 'break' or 'loop', so use the enclosing
                // loop scope
                let loop_scope = self.loop_scope.borrow();
                if loop_scope.get().len() == 0 {
                    self.tcx.sess.span_bug(sp, "break outside loop");
                } else {
                    // FIXME(#5275): this shouldn't have to be a method...
                    self.last_loop_scope()
                }
            }
        }
    }

    pub fn last_loop_scope(&self) -> NodeId {
        let loop_scope = self.loop_scope.borrow();
        *loop_scope.get().last().unwrap()
    }

    #[allow(unused_must_use)]
    pub fn ln_str(&self, ln: LiveNode) -> ~str {
        let mut wr = io::MemWriter::new();
        {
            let wr = &mut wr as &mut io::Writer;
            {
                let lnks = self.ir.lnks.try_borrow();
                write!(wr,
                       "[ln({}) of kind {:?} reads",
                       ln.get(),
                       lnks.and_then(|lnks| Some(lnks.get()[ln.get()])));
            }
            let users = self.users.try_borrow();
            match users {
                Some(users) => {
                    self.write_vars(wr, ln, |idx| users.get()[idx].reader);
                    write!(wr, "  writes");
                    self.write_vars(wr, ln, |idx| users.get()[idx].writer);
                }
                None => {
                    write!(wr, "  (users borrowed)");
                }
            }
            let successors = self.successors.try_borrow();
            match successors {
                Some(successors) => {
                    write!(wr, "  precedes {}]", successors.get()[ln.get()].to_str());
                }
                None => {
                    write!(wr, "  precedes (successors borrowed)]");
                }
            }
        }
        str::from_utf8_owned(wr.unwrap()).unwrap()
    }

    pub fn init_empty(&self, ln: LiveNode, succ_ln: LiveNode) {
        {
            let mut successors = self.successors.borrow_mut();
            successors.get()[ln.get()] = succ_ln;
        }

        // It is not necessary to initialize the
        // values to empty because this is the value
        // they have when they are created, and the sets
        // only grow during iterations.
        //
        // self.indices(ln) { |idx|
        //     self.users[idx] = invalid_users();
        // }
    }

    pub fn init_from_succ(&self, ln: LiveNode, succ_ln: LiveNode) {
        // more efficient version of init_empty() / merge_from_succ()
        {
            let mut successors = self.successors.borrow_mut();
            successors.get()[ln.get()] = succ_ln;
        }

        self.indices2(ln, succ_ln, |idx, succ_idx| {
            let mut users = self.users.borrow_mut();
            users.get()[idx] = users.get()[succ_idx]
        });
        debug!("init_from_succ(ln={}, succ={})",
               self.ln_str(ln), self.ln_str(succ_ln));
    }

    pub fn merge_from_succ(&self,
                           ln: LiveNode,
                           succ_ln: LiveNode,
                           first_merge: bool)
                           -> bool {
        if ln == succ_ln { return false; }

        let mut changed = false;
        self.indices2(ln, succ_ln, |idx, succ_idx| {
            let mut users = self.users.borrow_mut();
            changed |= copy_if_invalid(users.get()[succ_idx].reader,
                                       &mut users.get()[idx].reader);
            changed |= copy_if_invalid(users.get()[succ_idx].writer,
                                       &mut users.get()[idx].writer);
            if users.get()[succ_idx].used && !users.get()[idx].used {
                users.get()[idx].used = true;
                changed = true;
            }
        });

        debug!("merge_from_succ(ln={}, succ={}, first_merge={}, changed={})",
               ln.to_str(), self.ln_str(succ_ln), first_merge, changed);
        return changed;

        fn copy_if_invalid(src: LiveNode, dst: &mut LiveNode) -> bool {
            if src.is_valid() {
                if !dst.is_valid() {
                    *dst = src;
                    return true;
                }
            }
            return false;
        }
    }

    // Indicates that a local variable was *defined*; we know that no
    // uses of the variable can precede the definition (resolve checks
    // this) so we just clear out all the data.
    pub fn define(&self, writer: LiveNode, var: Variable) {
        let idx = self.idx(writer, var);
        let mut users = self.users.borrow_mut();
        users.get()[idx].reader = invalid_node();
        users.get()[idx].writer = invalid_node();

        debug!("{} defines {} (idx={}): {}", writer.to_str(), var.to_str(),
               idx, self.ln_str(writer));
    }

    // Either read, write, or both depending on the acc bitset
    pub fn acc(&self, ln: LiveNode, var: Variable, acc: uint) {
        let idx = self.idx(ln, var);
        let mut users = self.users.borrow_mut();
        let user = &mut users.get()[idx];

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

        debug!("{} accesses[{:x}] {}: {}",
               ln.to_str(), acc, var.to_str(), self.ln_str(ln));
    }

    // _______________________________________________________________________

    pub fn compute(&self, decl: &FnDecl, body: &Block) -> LiveNode {
        // if there is a `break` or `again` at the top level, then it's
        // effectively a return---this only occurs in `for` loops,
        // where the body is really a closure.

        debug!("compute: using id for block, {}", block_to_str(body,
                      self.tcx.sess.intr()));

        let entry_ln: LiveNode =
            self.with_loop_nodes(body.id, self.s.exit_ln, self.s.exit_ln,
              || { self.propagate_through_fn_block(decl, body) });

        // hack to skip the loop unless debug! is enabled:
        debug!("^^ liveness computation results for body {} (entry={})",
               {
                   for ln_idx in range(0u, self.ir.num_live_nodes.get()) {
                       debug!("{}", self.ln_str(LiveNode(ln_idx)));
                   }
                   body.id
               },
               entry_ln.to_str());

        entry_ln
    }

    pub fn propagate_through_fn_block(&self, _: &FnDecl, blk: &Block)
                                      -> LiveNode {
        // the fallthrough exit is only for those cases where we do not
        // explicitly return:
        self.init_from_succ(self.s.fallthrough_ln, self.s.exit_ln);
        if blk.expr.is_none() {
            self.acc(self.s.fallthrough_ln, self.s.no_ret_var, ACC_READ)
        }

        self.propagate_through_block(blk, self.s.fallthrough_ln)
    }

    pub fn propagate_through_block(&self, blk: &Block, succ: LiveNode)
                                   -> LiveNode {
        let succ = self.propagate_through_opt_expr(blk.expr, succ);
        blk.stmts.rev_iter().fold(succ, |succ, stmt| {
            self.propagate_through_stmt(*stmt, succ)
        })
    }

    pub fn propagate_through_stmt(&self, stmt: &Stmt, succ: LiveNode)
                                  -> LiveNode {
        match stmt.node {
          StmtDecl(decl, _) => {
            return self.propagate_through_decl(decl, succ);
          }

          StmtExpr(expr, _) | StmtSemi(expr, _) => {
            return self.propagate_through_expr(expr, succ);
          }

          StmtMac(..) => {
            self.tcx.sess.span_bug(stmt.span, "unexpanded macro");
          }
        }
    }

    pub fn propagate_through_decl(&self, decl: &Decl, succ: LiveNode)
                                  -> LiveNode {
        match decl.node {
            DeclLocal(ref local) => {
                self.propagate_through_local(*local, succ)
            }
            DeclItem(_) => succ,
        }
    }

    pub fn propagate_through_local(&self, local: &Local, succ: LiveNode)
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

        let succ = self.propagate_through_opt_expr(local.init, succ);
        self.define_bindings_in_pat(local.pat, succ)
    }

    pub fn propagate_through_exprs(&self, exprs: &[@Expr], succ: LiveNode)
                                   -> LiveNode {
        exprs.rev_iter().fold(succ, |succ, expr| {
            self.propagate_through_expr(*expr, succ)
        })
    }

    pub fn propagate_through_opt_expr(&self,
                                      opt_expr: Option<@Expr>,
                                      succ: LiveNode)
                                      -> LiveNode {
        opt_expr.iter().fold(succ, |succ, expr| {
            self.propagate_through_expr(*expr, succ)
        })
    }

    pub fn propagate_through_expr(&self, expr: @Expr, succ: LiveNode)
                                  -> LiveNode {
        debug!("propagate_through_expr: {}",
             expr_to_str(expr, self.tcx.sess.intr()));

        match expr.node {
          // Interesting cases with control flow or which gen/kill

          ExprPath(_) => {
              self.access_path(expr, succ, ACC_READ | ACC_USE)
          }

          ExprField(e, _, _) => {
              self.propagate_through_expr(e, succ)
          }

          ExprFnBlock(_, blk) | ExprProc(_, blk) => {
              debug!("{} is an ExprFnBlock or ExprProc",
                   expr_to_str(expr, self.tcx.sess.intr()));

              /*
              The next-node for a break is the successor of the entire
              loop. The next-node for a continue is the top of this loop.
              */
              self.with_loop_nodes(blk.id, succ,
                  self.live_node(expr.id, expr.span), || {

                 // the construction of a closure itself is not important,
                 // but we have to consider the closed over variables.
                 let caps = self.ir.captures(expr);
                 caps.rev_iter().fold(succ, |succ, cap| {
                     self.init_from_succ(cap.ln, succ);
                     let var = self.variable(cap.var_nid, expr.span);
                     self.acc(cap.ln, var, ACC_READ | ACC_USE);
                     cap.ln
                 })
              })
          }

          ExprIf(cond, then, els) => {
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
            let else_ln = self.propagate_through_opt_expr(els, succ);
            let then_ln = self.propagate_through_block(then, succ);
            let ln = self.live_node(expr.id, expr.span);
            self.init_from_succ(ln, else_ln);
            self.merge_from_succ(ln, then_ln, false);
            self.propagate_through_expr(cond, ln)
          }

          ExprWhile(cond, blk) => {
            self.propagate_through_loop(expr, Some(cond), blk, succ)
          }

          ExprForLoop(..) => fail!("non-desugared expr_for_loop"),

          // Note that labels have been resolved, so we don't need to look
          // at the label ident
          ExprLoop(blk, _) => {
            self.propagate_through_loop(expr, None, blk, succ)
          }

          ExprMatch(e, ref arms) => {
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
            for arm in arms.iter() {
                let body_succ =
                    self.propagate_through_block(arm.body, succ);
                let guard_succ =
                    self.propagate_through_opt_expr(arm.guard, body_succ);
                let arm_succ =
                    self.define_bindings_in_arm_pats(arm.pats, guard_succ);
                self.merge_from_succ(ln, arm_succ, first_merge);
                first_merge = false;
            };
            self.propagate_through_expr(e, ln)
          }

          ExprRet(o_e) => {
            // ignore succ and subst exit_ln:
            self.propagate_through_opt_expr(o_e, self.s.exit_ln)
          }

          ExprBreak(opt_label) => {
              // Find which label this break jumps to
              let sc = self.find_loop_scope(opt_label, expr.id, expr.span);

              // Now that we know the label we're going to,
              // look it up in the break loop nodes table

              let break_ln = self.break_ln.borrow();
              match break_ln.get().find(&sc) {
                  Some(&b) => b,
                  None => self.tcx.sess.span_bug(expr.span,
                                                 "break to unknown label")
              }
          }

          ExprAgain(opt_label) => {
              // Find which label this expr continues to
              let sc = self.find_loop_scope(opt_label, expr.id, expr.span);

              // Now that we know the label we're going to,
              // look it up in the continue loop nodes table

              let cont_ln = self.cont_ln.borrow();
              match cont_ln.get().find(&sc) {
                  Some(&b) => b,
                  None => self.tcx.sess.span_bug(expr.span,
                                                 "loop to unknown label")
              }
          }

          ExprAssign(l, r) => {
            // see comment on lvalues in
            // propagate_through_lvalue_components()
            let succ = self.write_lvalue(l, succ, ACC_WRITE);
            let succ = self.propagate_through_lvalue_components(l, succ);
            self.propagate_through_expr(r, succ)
          }

          ExprAssignOp(_, _, l, r) => {
            // see comment on lvalues in
            // propagate_through_lvalue_components()
            let succ = self.write_lvalue(l, succ, ACC_WRITE|ACC_READ);
            let succ = self.propagate_through_expr(r, succ);
            self.propagate_through_lvalue_components(l, succ)
          }

          // Uninteresting cases: just propagate in rev exec order

          ExprVstore(expr, _) => {
            self.propagate_through_expr(expr, succ)
          }

          ExprVec(ref exprs, _) => {
            self.propagate_through_exprs(*exprs, succ)
          }

          ExprRepeat(element, count, _) => {
            let succ = self.propagate_through_expr(count, succ);
            self.propagate_through_expr(element, succ)
          }

          ExprStruct(_, ref fields, with_expr) => {
            let succ = self.propagate_through_opt_expr(with_expr, succ);
            fields.rev_iter().fold(succ, |succ, field| {
                self.propagate_through_expr(field.expr, succ)
            })
          }

          ExprCall(f, ref args, _) => {
            // calling a fn with bot return type means that the fn
            // will fail, and hence the successors can be ignored
            let t_ret = ty::ty_fn_ret(ty::expr_ty(self.tcx, f));
            let succ = if ty::type_is_bot(t_ret) {self.s.exit_ln}
                       else {succ};
            let succ = self.propagate_through_exprs(*args, succ);
            self.propagate_through_expr(f, succ)
          }

          ExprMethodCall(callee_id, _, _, ref args, _) => {
            // calling a method with bot return type means that the method
            // will fail, and hence the successors can be ignored
            let t_ret = ty::ty_fn_ret(ty::node_id_to_type(self.tcx, callee_id));
            let succ = if ty::type_is_bot(t_ret) {self.s.exit_ln}
                       else {succ};
            self.propagate_through_exprs(*args, succ)
          }

          ExprTup(ref exprs) => {
            self.propagate_through_exprs(*exprs, succ)
          }

          ExprBinary(_, op, l, r) if ast_util::lazy_binop(op) => {
            let r_succ = self.propagate_through_expr(r, succ);

            let ln = self.live_node(expr.id, expr.span);
            self.init_from_succ(ln, succ);
            self.merge_from_succ(ln, r_succ, false);

            self.propagate_through_expr(l, ln)
          }

          ExprIndex(_, l, r) |
          ExprBinary(_, _, l, r) |
          ExprBox(l, r) => {
            self.propagate_through_exprs([l, r], succ)
          }

          ExprAddrOf(_, e) |
          ExprCast(e, _) |
          ExprUnary(_, _, e) |
          ExprParen(e) => {
            self.propagate_through_expr(e, succ)
          }

          ExprInlineAsm(ref ia) => {
            let succ = ia.inputs.rev_iter().fold(succ, |succ, &(_, expr)| {
                self.propagate_through_expr(expr, succ)
            });
            ia.outputs.rev_iter().fold(succ, |succ, &(_, expr)| {
                // see comment on lvalues in
                // propagate_through_lvalue_components()
                let succ = self.write_lvalue(expr, succ, ACC_WRITE);
                self.propagate_through_lvalue_components(expr, succ)
            })
          }

          ExprLogLevel |
          ExprLit(..) => {
            succ
          }

          ExprBlock(blk) => {
            self.propagate_through_block(blk, succ)
          }

          ExprMac(..) => {
            self.tcx.sess.span_bug(expr.span, "unexpanded macro");
          }
        }
    }

    pub fn propagate_through_lvalue_components(&self,
                                               expr: @Expr,
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
            ExprPath(_) => succ,
            ExprField(e, _, _) => self.propagate_through_expr(e, succ),
            _ => self.propagate_through_expr(expr, succ)
        }
    }

    // see comment on propagate_through_lvalue()
    pub fn write_lvalue(&self, expr: &Expr, succ: LiveNode, acc: uint)
                        -> LiveNode {
        match expr.node {
          ExprPath(_) => self.access_path(expr, succ, acc),

          // We do not track other lvalues, so just propagate through
          // to their subcomponents.  Also, it may happen that
          // non-lvalues occur here, because those are detected in the
          // later pass borrowck.
          _ => succ
        }
    }

    pub fn access_path(&self, expr: &Expr, succ: LiveNode, acc: uint)
                       -> LiveNode {
        let def_map = self.tcx.def_map.borrow();
        let def = def_map.get().get_copy(&expr.id);
        match moves::moved_variable_node_id_from_def(def) {
          Some(nid) => {
            let ln = self.live_node(expr.id, expr.span);
            if acc != 0u {
                self.init_from_succ(ln, succ);
                let var = self.variable(nid, expr.span);
                self.acc(ln, var, acc);
            }
            ln
          }
          None => succ
        }
    }

    pub fn propagate_through_loop(&self,
                                  expr: &Expr,
                                  cond: Option<@Expr>,
                                  body: &Block,
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
        if cond.is_some() {
            // if there is a condition, then it's possible we bypass
            // the body altogether.  otherwise, the only way is via a
            // break in the loop body.
            self.merge_from_succ(ln, succ, first_merge);
            first_merge = false;
        }
        debug!("propagate_through_loop: using id for loop body {} {}",
               expr.id, block_to_str(body, self.tcx.sess.intr()));

        let cond_ln = self.propagate_through_opt_expr(cond, ln);
        let body_ln = self.with_loop_nodes(expr.id, succ, ln, || {
            self.propagate_through_block(body, cond_ln)
        });

        // repeat until fixed point is reached:
        while self.merge_from_succ(ln, body_ln, first_merge) {
            first_merge = false;
            assert!(cond_ln == self.propagate_through_opt_expr(cond,
                                                                    ln));
            assert!(body_ln == self.with_loop_nodes(expr.id, succ, ln,
            || {
                self.propagate_through_block(body, cond_ln)
            }));
        }

        cond_ln
    }

    pub fn with_loop_nodes<R>(
                           &self,
                           loop_node_id: NodeId,
                           break_ln: LiveNode,
                           cont_ln: LiveNode,
                           f: || -> R)
                           -> R {
        debug!("with_loop_nodes: {} {}", loop_node_id, break_ln.get());
        {
            let mut loop_scope = self.loop_scope.borrow_mut();
            loop_scope.get().push(loop_node_id);
        }
        {
            let mut this_break_ln = self.break_ln.borrow_mut();
            let mut this_cont_ln = self.cont_ln.borrow_mut();
            this_break_ln.get().insert(loop_node_id, break_ln);
            this_cont_ln.get().insert(loop_node_id, cont_ln);
        }
        let r = f();
        {
            let mut loop_scope = self.loop_scope.borrow_mut();
            loop_scope.get().pop();
        }
        r
    }
}

// _______________________________________________________________________
// Checking for error conditions

fn check_local(this: &mut Liveness, local: &Local) {
    match local.init {
      Some(_) => {
        this.warn_about_unused_or_dead_vars_in_pat(local.pat);
      }
      None => {

        // No initializer: the variable might be unused; if not, it
        // should not be live at this point.

        debug!("check_local() with no initializer");
        this.pat_bindings(local.pat, |ln, var, sp, id| {
            if !this.warn_about_unused(sp, id, ln, var) {
                match this.live_on_exit(ln, var) {
                  None => { /* not live: good */ }
                  Some(lnk) => {
                    this.report_illegal_read(
                        local.span, lnk, var,
                        PossiblyUninitializedVariable);
                  }
                }
            }
        })
      }
    }

    visit::walk_local(this, local, ());
}

fn check_arm(this: &mut Liveness, arm: &Arm) {
    this.arm_pats_bindings(arm.pats, |ln, var, sp, id| {
        this.warn_about_unused(sp, id, ln, var);
    });
    visit::walk_arm(this, arm, ());
}

fn check_expr(this: &mut Liveness, expr: &Expr) {
    match expr.node {
      ExprAssign(l, r) => {
        this.check_lvalue(l);
        this.visit_expr(r, ());

        visit::walk_expr(this, expr, ());
      }

      ExprAssignOp(_, _, l, _) => {
        this.check_lvalue(l);

        visit::walk_expr(this, expr, ());
      }

      ExprInlineAsm(ref ia) => {
        for &(_, input) in ia.inputs.iter() {
          this.visit_expr(input, ());
        }

        // Output operands must be lvalues
        for &(_, out) in ia.outputs.iter() {
          this.check_lvalue(out);
          this.visit_expr(out, ());
        }

        visit::walk_expr(this, expr, ());
      }

      // no correctness conditions related to liveness
      ExprCall(..) | ExprMethodCall(..) | ExprIf(..) | ExprMatch(..) |
      ExprWhile(..) | ExprLoop(..) | ExprIndex(..) | ExprField(..) |
      ExprVstore(..) | ExprVec(..) | ExprTup(..) | ExprLogLevel |
      ExprBinary(..) |
      ExprCast(..) | ExprUnary(..) | ExprRet(..) | ExprBreak(..) |
      ExprAgain(..) | ExprLit(_) | ExprBlock(..) |
      ExprMac(..) | ExprAddrOf(..) | ExprStruct(..) | ExprRepeat(..) |
      ExprParen(..) | ExprFnBlock(..) | ExprProc(..) | ExprPath(..) |
      ExprBox(..) => {
        visit::walk_expr(this, expr, ());
      }
      ExprForLoop(..) => fail!("non-desugared expr_for_loop")
    }
}

fn check_fn(_v: &Liveness,
            _fk: &FnKind,
            _decl: &FnDecl,
            _body: &Block,
            _sp: Span,
            _id: NodeId) {
    // do not check contents of nested fns
}

enum ReadKind {
    PossiblyUninitializedVariable,
    PossiblyUninitializedField,
    MovedValue,
    PartiallyMovedValue
}

impl Liveness {
    pub fn check_ret(&self,
                     id: NodeId,
                     sp: Span,
                     _fk: &FnKind,
                     entry_ln: LiveNode,
                     body: &Block) {
        if self.live_on_entry(entry_ln, self.s.no_ret_var).is_some() {
            // if no_ret_var is live, then we fall off the end of the
            // function without any kind of return expression:

            let t_ret = ty::ty_fn_ret(ty::node_id_to_type(self.tcx, id));
            if ty::type_is_nil(t_ret) {
                // for nil return types, it is ok to not return a value expl.
            } else if ty::type_is_bot(t_ret) {
                // for bot return types, not ok.  Function should fail.
                self.tcx.sess.span_err(
                    sp, "some control paths may return");
            } else {
                let ends_with_stmt = match body.expr {
                    None if body.stmts.len() > 0 =>
                        match body.stmts.last().unwrap().node {
                            StmtSemi(e, _) => {
                                let t_stmt = ty::expr_ty(self.tcx, e);
                                ty::get(t_stmt).sty == ty::get(t_ret).sty
                            },
                            _ => false
                        },
                    _ => false
                };
                if ends_with_stmt {
                    let last_stmt = body.stmts.last().unwrap();
                    let span_semicolon = Span {
                        lo: last_stmt.span.hi,
                        hi: last_stmt.span.hi,
                        expn_info: last_stmt.span.expn_info
                    };
                    self.tcx.sess.span_note(
                        span_semicolon, "consider removing this semicolon:");
                }
                self.tcx.sess.span_err(
                    sp, "not all control paths return a value");
           }
        }
    }

    pub fn check_lvalue(&mut self, expr: @Expr) {
        match expr.node {
          ExprPath(_) => {
            let def_map = self.tcx.def_map.borrow();
            match def_map.get().get_copy(&expr.id) {
              DefLocal(nid, _) => {
                // Assignment to an immutable variable or argument: only legal
                // if there is no later assignment. If this local is actually
                // mutable, then check for a reassignment to flag the mutability
                // as being used.
                let ln = self.live_node(expr.id, expr.span);
                let var = self.variable(nid, expr.span);
                self.warn_about_dead_assign(expr.span, expr.id, ln, var);
              }
              def => {
                match moves::moved_variable_node_id_from_def(def) {
                  Some(nid) => {
                    let ln = self.live_node(expr.id, expr.span);
                    let var = self.variable(nid, expr.span);
                    self.warn_about_dead_assign(expr.span, expr.id, ln, var);
                  }
                  None => {}
                }
              }
            }
          }

          _ => {
            // For other kinds of lvalues, no checks are required,
            // and any embedded expressions are actually rvalues
            visit::walk_expr(self, expr, ());
          }
       }
    }

    pub fn report_illegal_read(&self,
                               chk_span: Span,
                               lnk: LiveNodeKind,
                               var: Variable,
                               rk: ReadKind) {
        let msg = match rk {
            PossiblyUninitializedVariable => "possibly uninitialized \
                                              variable",
            PossiblyUninitializedField => "possibly uninitialized field",
            MovedValue => "moved value",
            PartiallyMovedValue => "partially moved value"
        };
        let name = self.ir.variable_name(var);
        match lnk {
          FreeVarNode(span) => {
            self.tcx.sess.span_err(
                span,
                format!("capture of {}: `{}`", msg, name));
          }
          ExprNode(span) => {
            self.tcx.sess.span_err(
                span,
                format!("use of {}: `{}`", msg, name));
          }
          ExitNode | VarDefNode(_) => {
            self.tcx.sess.span_bug(
                chk_span,
                format!("illegal reader: {:?}", lnk));
          }
        }
    }

    pub fn should_warn(&self, var: Variable) -> Option<~str> {
        let name = self.ir.variable_name(var);
        if name.len() == 0 || name[0] == ('_' as u8) { None } else { Some(name) }
    }

    pub fn warn_about_unused_args(&self, decl: &FnDecl, entry_ln: LiveNode) {
        for arg in decl.inputs.iter() {
            pat_util::pat_bindings(self.tcx.def_map,
                                   arg.pat,
                                   |_bm, p_id, sp, path| {
                let var = self.variable(p_id, sp);
                // Ignore unused self.
                let ident = ast_util::path_to_ident(path);
                if ident.name != special_idents::self_.name {
                    self.warn_about_unused(sp, p_id, entry_ln, var);
                }
            })
        }
    }

    pub fn warn_about_unused_or_dead_vars_in_pat(&self, pat: @Pat) {
        self.pat_bindings(pat, |ln, var, sp, id| {
            if !self.warn_about_unused(sp, id, ln, var) {
                self.warn_about_dead_assign(sp, id, ln, var);
            }
        })
    }

    pub fn warn_about_unused(&self,
                             sp: Span,
                             id: NodeId,
                             ln: LiveNode,
                             var: Variable)
                             -> bool {
        if !self.used_on_entry(ln, var) {
            let r = self.should_warn(var);
            for name in r.iter() {

                // annoying: for parameters in funcs like `fn(x: int)
                // {ret}`, there is only one node, so asking about
                // assigned_on_exit() is not meaningful.
                let is_assigned = if ln == self.s.exit_ln {
                    false
                } else {
                    self.assigned_on_exit(ln, var).is_some()
                };

                if is_assigned {
                    self.tcx.sess.add_lint(UnusedVariable, id, sp,
                        format!("variable `{}` is assigned to, \
                                  but never used", *name));
                } else {
                    self.tcx.sess.add_lint(UnusedVariable, id, sp,
                        format!("unused variable: `{}`", *name));
                }
            }
            true
        } else {
            false
        }
    }

    pub fn warn_about_dead_assign(&self,
                                  sp: Span,
                                  id: NodeId,
                                  ln: LiveNode,
                                  var: Variable) {
        if self.live_on_exit(ln, var).is_none() {
            let r = self.should_warn(var);
            for name in r.iter() {
                self.tcx.sess.add_lint(DeadAssignment, id, sp,
                    format!("value assigned to `{}` is never read", *name));
            }
        }
    }
 }
