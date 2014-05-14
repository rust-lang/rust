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


use middle::freevars;
use middle::lint::{UnusedVariable, DeadAssignment};
use middle::pat_util;
use middle::ty;
use util::nodemap::NodeMap;

use std::mem::transmute;
use std::fmt;
use std::io;
use std::rc::Rc;
use std::str;
use std::uint;
use syntax::ast::*;
use syntax::codemap::{BytePos, original_sp, Span};
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

fn live_node_kind_to_str(lnk: LiveNodeKind, cx: &ty::ctxt) -> StrBuf {
    let cm = cx.sess.codemap();
    match lnk {
        FreeVarNode(s) => {
            format_strbuf!("Free var node [{}]", cm.span_to_str(s))
        }
        ExprNode(s) => {
            format_strbuf!("Expr node [{}]", cm.span_to_str(s))
        }
        VarDefNode(s) => {
            format_strbuf!("Var def node [{}]", cm.span_to_str(s))
        }
        ExitNode => "Exit node".to_strbuf(),
    }
}

impl<'a> Visitor<()> for IrMaps<'a> {
    fn visit_fn(&mut self, fk: &FnKind, fd: &FnDecl, b: &Block, s: Span, n: NodeId, _: ()) {
        visit_fn(self, fk, fd, b, s, n);
    }
    fn visit_local(&mut self, l: &Local, _: ()) { visit_local(self, l); }
    fn visit_expr(&mut self, ex: &Expr, _: ()) { visit_expr(self, ex); }
    fn visit_arm(&mut self, a: &Arm, _: ()) { visit_arm(self, a); }
}

pub fn check_crate(tcx: &ty::ctxt,
                   krate: &Crate) {
    visit::walk_crate(&mut IrMaps(tcx), krate, ());
    tcx.sess.abort_if_errors();
}

impl fmt::Show for LiveNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f.buf, "ln({})", self.get())
    }
}

impl fmt::Show for Variable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f.buf, "v({})", self.get())
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

struct IrMaps<'a> {
    tcx: &'a ty::ctxt,

    num_live_nodes: uint,
    num_vars: uint,
    live_node_map: NodeMap<LiveNode>,
    variable_map: NodeMap<Variable>,
    capture_info_map: NodeMap<Rc<Vec<CaptureInfo>>>,
    var_kinds: Vec<VarKind>,
    lnks: Vec<LiveNodeKind>,
}

fn IrMaps<'a>(tcx: &'a ty::ctxt)
              -> IrMaps<'a> {
    IrMaps {
        tcx: tcx,
        num_live_nodes: 0,
        num_vars: 0,
        live_node_map: NodeMap::new(),
        variable_map: NodeMap::new(),
        capture_info_map: NodeMap::new(),
        var_kinds: Vec::new(),
        lnks: Vec::new(),
    }
}

impl<'a> IrMaps<'a> {
    fn add_live_node(&mut self, lnk: LiveNodeKind) -> LiveNode {
        let ln = LiveNode(self.num_live_nodes);
        self.lnks.push(lnk);
        self.num_live_nodes += 1;

        debug!("{} is of kind {}", ln.to_str(),
               live_node_kind_to_str(lnk, self.tcx));

        ln
    }

    fn add_live_node_for_node(&mut self, node_id: NodeId, lnk: LiveNodeKind) {
        let ln = self.add_live_node(lnk);
        self.live_node_map.insert(node_id, ln);

        debug!("{} is node {}", ln.to_str(), node_id);
    }

    fn add_variable(&mut self, vk: VarKind) -> Variable {
        let v = Variable(self.num_vars);
        self.var_kinds.push(vk);
        self.num_vars += 1;

        match vk {
            Local(LocalInfo { id: node_id, .. }) | Arg(node_id, _) => {
                self.variable_map.insert(node_id, v);
            },
            ImplicitRet => {}
        }

        debug!("{} is {:?}", v.to_str(), vk);

        v
    }

    fn variable(&self, node_id: NodeId, span: Span) -> Variable {
        match self.variable_map.find(&node_id) {
          Some(&var) => var,
          None => {
            self.tcx
                .sess
                .span_bug(span, format!("no variable registered for id {}",
                                        node_id));
          }
        }
    }

    fn variable_name(&self, var: Variable) -> StrBuf {
        match self.var_kinds.get(var.get()) {
            &Local(LocalInfo { ident: nm, .. }) | &Arg(_, nm) => {
                token::get_ident(nm).get().to_str().to_strbuf()
            },
            &ImplicitRet => "<implicit-ret>".to_strbuf()
        }
    }

    fn set_captures(&mut self, node_id: NodeId, cs: Vec<CaptureInfo>) {
        self.capture_info_map.insert(node_id, Rc::new(cs));
    }

    fn lnk(&self, ln: LiveNode) -> LiveNodeKind {
        *self.lnks.get(ln.get())
    }
}

impl<'a> Visitor<()> for Liveness<'a> {
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

fn visit_fn(ir: &mut IrMaps,
            fk: &FnKind,
            decl: &FnDecl,
            body: &Block,
            sp: Span,
            id: NodeId) {
    debug!("visit_fn: id={}", id);
    let _i = ::util::common::indenter();

    // swap in a new set of IR maps for this function body:
    let mut fn_maps = IrMaps(ir.tcx);

    unsafe {
        debug!("creating fn_maps: {}", transmute::<&IrMaps, *IrMaps>(&fn_maps));
    }

    for arg in decl.inputs.iter() {
        pat_util::pat_bindings(&ir.tcx.def_map,
                               arg.pat,
                               |_bm, arg_id, _x, path| {
            debug!("adding argument {}", arg_id);
            let ident = ast_util::path_to_ident(path);
            fn_maps.add_variable(Arg(arg_id, ident));
        })
    };

    // gather up the various local variables, significant expressions,
    // and so forth:
    visit::walk_fn(&mut fn_maps, fk, decl, body, sp, ());

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
    let mut lsets = Liveness(&mut fn_maps, specials);
    let entry_ln = lsets.compute(decl, body);

    // check for various error conditions
    lsets.visit_block(body, ());
    lsets.check_ret(id, sp, fk, entry_ln, body);
    lsets.warn_about_unused_args(decl, entry_ln);
}

fn visit_local(ir: &mut IrMaps, local: &Local) {
    pat_util::pat_bindings(&ir.tcx.def_map, local.pat, |bm, p_id, sp, path| {
        debug!("adding local variable {}", p_id);
        let name = ast_util::path_to_ident(path);
        ir.add_live_node_for_node(p_id, VarDefNode(sp));
        let kind = match local.init {
          Some(_) => FromLetWithInitializer,
          None => FromLetNoInitializer
        };
        let mutbl = match bm {
            BindByValue(MutMutable) => true,
            _ => false
        };
        ir.add_variable(Local(LocalInfo {
          id: p_id,
          ident: name,
          is_mutbl: mutbl,
          kind: kind
        }));
    });
    visit::walk_local(ir, local, ());
}

fn visit_arm(ir: &mut IrMaps, arm: &Arm) {
    for pat in arm.pats.iter() {
        pat_util::pat_bindings(&ir.tcx.def_map, *pat, |bm, p_id, sp, path| {
            debug!("adding local variable {} from match with bm {:?}",
                   p_id, bm);
            let name = ast_util::path_to_ident(path);
            let mutbl = match bm {
                BindByValue(MutMutable) => true,
                _ => false
            };
            ir.add_live_node_for_node(p_id, VarDefNode(sp));
            ir.add_variable(Local(LocalInfo {
                id: p_id,
                ident: name,
                is_mutbl: mutbl,
                kind: FromMatch(bm)
            }));
        })
    }
    visit::walk_arm(ir, arm, ());
}

fn moved_variable_node_id_from_def(def: Def) -> Option<NodeId> {
    match def {
        DefBinding(nid, _) |
        DefArg(nid, _) |
        DefLocal(nid, _) => Some(nid),

      _ => None
    }
}

fn visit_expr(ir: &mut IrMaps, expr: &Expr) {
    match expr.node {
      // live nodes required for uses or definitions of variables:
      ExprPath(_) => {
        let def = ir.tcx.def_map.borrow().get_copy(&expr.id);
        debug!("expr {}: path that leads to {:?}", expr.id, def);
        if moved_variable_node_id_from_def(def).is_some() {
            ir.add_live_node_for_node(expr.id, ExprNode(expr.span));
        }
        visit::walk_expr(ir, expr, ());
      }
      ExprFnBlock(..) | ExprProc(..) => {
        // Interesting control flow (for loops can contain labeled
        // breaks or continues)
        ir.add_live_node_for_node(expr.id, ExprNode(expr.span));

        // Make a live_node for each captured variable, with the span
        // being the location that the variable is used.  This results
        // in better error messages than just pointing at the closure
        // construction site.
        let mut call_caps = Vec::new();
        let fv_mode = freevars::get_capture_mode(ir.tcx, expr.id);
        freevars::with_freevars(ir.tcx, expr.id, |freevars| {
            for fv in freevars.iter() {
                match moved_variable_node_id_from_def(fv.def) {
                    Some(rv) => {
                        let fv_ln = ir.add_live_node(FreeVarNode(fv.span));
                        let fv_id = ast_util::def_id_of_def(fv.def).node;
                        let fv_ty = ty::node_id_to_type(ir.tcx, fv_id);
                        let is_move = match fv_mode {
                            // var must be dead afterwards
                            freevars::CaptureByValue => {
                                ty::type_moves_by_default(ir.tcx, fv_ty)
                            }

                            // var can still be used
                            freevars::CaptureByRef => {
                                false
                            }
                        };
                        call_caps.push(CaptureInfo {ln: fv_ln,
                                                    is_move: is_move,
                                                    var_nid: rv});
                    }
                    None => {}
                }
            }
        });
        ir.set_captures(expr.id, call_caps);

        visit::walk_expr(ir, expr, ());
      }

      // live nodes required for interesting control flow:
      ExprIf(..) | ExprMatch(..) | ExprWhile(..) | ExprLoop(..) => {
        ir.add_live_node_for_node(expr.id, ExprNode(expr.span));
        visit::walk_expr(ir, expr, ());
      }
      ExprForLoop(..) => fail!("non-desugared expr_for_loop"),
      ExprBinary(op, _, _) if ast_util::lazy_binop(op) => {
        ir.add_live_node_for_node(expr.id, ExprNode(expr.span));
        visit::walk_expr(ir, expr, ());
      }

      // otherwise, live nodes are not required:
      ExprIndex(..) | ExprField(..) | ExprVstore(..) | ExprVec(..) |
      ExprCall(..) | ExprMethodCall(..) | ExprTup(..) |
      ExprBinary(..) | ExprAddrOf(..) |
      ExprCast(..) | ExprUnary(..) | ExprBreak(_) |
      ExprAgain(_) | ExprLit(_) | ExprRet(..) | ExprBlock(..) |
      ExprAssign(..) | ExprAssignOp(..) | ExprMac(..) |
      ExprStruct(..) | ExprRepeat(..) | ExprParen(..) |
      ExprInlineAsm(..) | ExprBox(..) => {
          visit::walk_expr(ir, expr, ());
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

struct Liveness<'a> {
    ir: &'a mut IrMaps<'a>,
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

fn Liveness<'a>(ir: &'a mut IrMaps<'a>, specials: Specials) -> Liveness<'a> {
    Liveness {
        ir: ir,
        s: specials,
        successors: Vec::from_elem(ir.num_live_nodes, invalid_node()),
        users: Vec::from_elem(ir.num_live_nodes * ir.num_vars, invalid_users()),
        loop_scope: Vec::new(),
        break_ln: NodeMap::new(),
        cont_ln: NodeMap::new(),
    }
}

impl<'a> Liveness<'a> {
    fn live_node(&self, node_id: NodeId, span: Span) -> LiveNode {
        match self.ir.live_node_map.find(&node_id) {
          Some(&ln) => ln,
          None => {
            // This must be a mismatch between the ir_map construction
            // above and the propagation code below; the two sets of
            // code have to agree about which AST nodes are worth
            // creating liveness nodes for.
            self.ir.tcx.sess.span_bug(
                span, format!("no live node registered for node {}",
                           node_id));
          }
        }
    }

    fn variable(&self, node_id: NodeId, span: Span) -> Variable {
        self.ir.variable(node_id, span)
    }

    fn pat_bindings(&mut self,
                    pat: &Pat,
                    f: |&mut Liveness<'a>, LiveNode, Variable, Span, NodeId|) {
        pat_util::pat_bindings(&self.ir.tcx.def_map, pat, |_bm, p_id, sp, _n| {
            let ln = self.live_node(p_id, sp);
            let var = self.variable(p_id, sp);
            f(self, ln, var, sp, p_id);
        })
    }

    fn arm_pats_bindings(&mut self,
                         pats: &[@Pat],
                         f: |&mut Liveness<'a>, LiveNode, Variable, Span, NodeId|) {
        // only consider the first pattern; any later patterns must have
        // the same bindings, and we also consider the first pattern to be
        // the "authoritative" set of ids
        if !pats.is_empty() {
            self.pat_bindings(pats[0], f)
        }
    }

    fn define_bindings_in_pat(&mut self, pat: @Pat, succ: LiveNode)
                              -> LiveNode {
        self.define_bindings_in_arm_pats([pat], succ)
    }

    fn define_bindings_in_arm_pats(&mut self, pats: &[@Pat], succ: LiveNode)
                                   -> LiveNode {
        let mut succ = succ;
        self.arm_pats_bindings(pats, |this, ln, var, _sp, _id| {
            this.init_from_succ(ln, succ);
            this.define(ln, var);
            succ = ln;
        });
        succ
    }

    fn idx(&self, ln: LiveNode, var: Variable) -> uint {
        ln.get() * self.ir.num_vars + var.get()
    }

    fn live_on_entry(&self, ln: LiveNode, var: Variable)
                      -> Option<LiveNodeKind> {
        assert!(ln.is_valid());
        let reader = self.users.get(self.idx(ln, var)).reader;
        if reader.is_valid() {Some(self.ir.lnk(reader))} else {None}
    }

    /*
    Is this variable live on entry to any of its successor nodes?
    */
    fn live_on_exit(&self, ln: LiveNode, var: Variable)
                    -> Option<LiveNodeKind> {
        let successor = *self.successors.get(ln.get());
        self.live_on_entry(successor, var)
    }

    fn used_on_entry(&self, ln: LiveNode, var: Variable) -> bool {
        assert!(ln.is_valid());
        self.users.get(self.idx(ln, var)).used
    }

    fn assigned_on_entry(&self, ln: LiveNode, var: Variable)
                         -> Option<LiveNodeKind> {
        assert!(ln.is_valid());
        let writer = self.users.get(self.idx(ln, var)).writer;
        if writer.is_valid() {Some(self.ir.lnk(writer))} else {None}
    }

    fn assigned_on_exit(&self, ln: LiveNode, var: Variable)
                        -> Option<LiveNodeKind> {
        let successor = *self.successors.get(ln.get());
        self.assigned_on_entry(successor, var)
    }

    fn indices2(&mut self,
                ln: LiveNode,
                succ_ln: LiveNode,
                op: |&mut Liveness<'a>, uint, uint|) {
        let node_base_idx = self.idx(ln, Variable(0u));
        let succ_base_idx = self.idx(succ_ln, Variable(0u));
        for var_idx in range(0u, self.ir.num_vars) {
            op(self, node_base_idx + var_idx, succ_base_idx + var_idx);
        }
    }

    fn write_vars(&self,
                  wr: &mut io::Writer,
                  ln: LiveNode,
                  test: |uint| -> LiveNode) -> io::IoResult<()> {
        let node_base_idx = self.idx(ln, Variable(0));
        for var_idx in range(0u, self.ir.num_vars) {
            let idx = node_base_idx + var_idx;
            if test(idx).is_valid() {
                try!(write!(wr, " {}", Variable(var_idx).to_str()));
            }
        }
        Ok(())
    }

    fn find_loop_scope(&self,
                       opt_label: Option<Ident>,
                       id: NodeId,
                       sp: Span)
                       -> NodeId {
        match opt_label {
            Some(_) => {
                // Refers to a labeled loop. Use the results of resolve
                // to find with one
                match self.ir.tcx.def_map.borrow().find(&id) {
                    Some(&DefLabel(loop_id)) => loop_id,
                    _ => self.ir.tcx.sess.span_bug(sp, "label on break/loop \
                                                        doesn't refer to a loop")
                }
            }
            None => {
                // Vanilla 'break' or 'loop', so use the enclosing
                // loop scope
                if self.loop_scope.len() == 0 {
                    self.ir.tcx.sess.span_bug(sp, "break outside loop");
                } else {
                    // FIXME(#5275): this shouldn't have to be a method...
                    self.last_loop_scope()
                }
            }
        }
    }

    fn last_loop_scope(&self) -> NodeId {
        *self.loop_scope.last().unwrap()
    }

    #[allow(unused_must_use)]
    fn ln_str(&self, ln: LiveNode) -> StrBuf {
        let mut wr = io::MemWriter::new();
        {
            let wr = &mut wr as &mut io::Writer;
            write!(wr, "[ln({}) of kind {:?} reads", ln.get(), self.ir.lnk(ln));
            self.write_vars(wr, ln, |idx| self.users.get(idx).reader);
            write!(wr, "  writes");
            self.write_vars(wr, ln, |idx| self.users.get(idx).writer);
            write!(wr, "  precedes {}]", self.successors.get(ln.get()).to_str());
        }
        str::from_utf8(wr.unwrap().as_slice()).unwrap().to_strbuf()
    }

    fn init_empty(&mut self, ln: LiveNode, succ_ln: LiveNode) {
        *self.successors.get_mut(ln.get()) = succ_ln;

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
        *self.successors.get_mut(ln.get()) = succ_ln;

        self.indices2(ln, succ_ln, |this, idx, succ_idx| {
            *this.users.get_mut(idx) = *this.users.get(succ_idx)
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
            changed |= copy_if_invalid(this.users.get(succ_idx).reader,
                                       &mut this.users.get_mut(idx).reader);
            changed |= copy_if_invalid(this.users.get(succ_idx).writer,
                                       &mut this.users.get_mut(idx).writer);
            if this.users.get(succ_idx).used && !this.users.get(idx).used {
                this.users.get_mut(idx).used = true;
                changed = true;
            }
        });

        debug!("merge_from_succ(ln={}, succ={}, first_merge={}, changed={})",
               ln.to_str(), self.ln_str(succ_ln), first_merge, changed);
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
        self.users.get_mut(idx).reader = invalid_node();
        self.users.get_mut(idx).writer = invalid_node();

        debug!("{} defines {} (idx={}): {}", writer.to_str(), var.to_str(),
               idx, self.ln_str(writer));
    }

    // Either read, write, or both depending on the acc bitset
    fn acc(&mut self, ln: LiveNode, var: Variable, acc: uint) {
        debug!("{} accesses[{:x}] {}: {}",
               ln.to_str(), acc, var.to_str(), self.ln_str(ln));

        let idx = self.idx(ln, var);
        let user = self.users.get_mut(idx);

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

    fn compute(&mut self, decl: &FnDecl, body: &Block) -> LiveNode {
        // if there is a `break` or `again` at the top level, then it's
        // effectively a return---this only occurs in `for` loops,
        // where the body is really a closure.

        debug!("compute: using id for block, {}", block_to_str(body));

        let entry_ln: LiveNode =
            self.with_loop_nodes(body.id, self.s.exit_ln, self.s.exit_ln,
              |this| this.propagate_through_fn_block(decl, body));

        // hack to skip the loop unless debug! is enabled:
        debug!("^^ liveness computation results for body {} (entry={})",
               {
                   for ln_idx in range(0u, self.ir.num_live_nodes) {
                       debug!("{}", self.ln_str(LiveNode(ln_idx)));
                   }
                   body.id
               },
               entry_ln.to_str());

        entry_ln
    }

    fn propagate_through_fn_block(&mut self, _: &FnDecl, blk: &Block)
                                  -> LiveNode {
        // the fallthrough exit is only for those cases where we do not
        // explicitly return:
        self.init_from_succ(self.s.fallthrough_ln, self.s.exit_ln);
        if blk.expr.is_none() {
            self.acc(self.s.fallthrough_ln, self.s.no_ret_var, ACC_READ)
        }

        self.propagate_through_block(blk, self.s.fallthrough_ln)
    }

    fn propagate_through_block(&mut self, blk: &Block, succ: LiveNode)
                               -> LiveNode {
        let succ = self.propagate_through_opt_expr(blk.expr, succ);
        blk.stmts.iter().rev().fold(succ, |succ, stmt| {
            self.propagate_through_stmt(*stmt, succ)
        })
    }

    fn propagate_through_stmt(&mut self, stmt: &Stmt, succ: LiveNode)
                              -> LiveNode {
        match stmt.node {
            StmtDecl(decl, _) => {
                self.propagate_through_decl(decl, succ)
            }

            StmtExpr(expr, _) | StmtSemi(expr, _) => {
                self.propagate_through_expr(expr, succ)
            }

            StmtMac(..) => {
                self.ir.tcx.sess.span_bug(stmt.span, "unexpanded macro");
            }
        }
    }

    fn propagate_through_decl(&mut self, decl: &Decl, succ: LiveNode)
                              -> LiveNode {
        match decl.node {
            DeclLocal(ref local) => {
                self.propagate_through_local(*local, succ)
            }
            DeclItem(_) => succ,
        }
    }

    fn propagate_through_local(&mut self, local: &Local, succ: LiveNode)
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

    fn propagate_through_exprs(&mut self, exprs: &[@Expr], succ: LiveNode)
                               -> LiveNode {
        exprs.iter().rev().fold(succ, |succ, expr| {
            self.propagate_through_expr(*expr, succ)
        })
    }

    fn propagate_through_opt_expr(&mut self,
                                  opt_expr: Option<@Expr>,
                                  succ: LiveNode)
                                  -> LiveNode {
        opt_expr.iter().fold(succ, |succ, expr| {
            self.propagate_through_expr(*expr, succ)
        })
    }

    fn propagate_through_expr(&mut self, expr: &Expr, succ: LiveNode)
                              -> LiveNode {
        debug!("propagate_through_expr: {}", expr_to_str(expr));

        match expr.node {
          // Interesting cases with control flow or which gen/kill

          ExprPath(_) => {
              self.access_path(expr, succ, ACC_READ | ACC_USE)
          }

          ExprField(e, _, _) => {
              self.propagate_through_expr(e, succ)
          }

          ExprFnBlock(_, blk) | ExprProc(_, blk) => {
              debug!("{} is an ExprFnBlock or ExprProc", expr_to_str(expr));

              /*
              The next-node for a break is the successor of the entire
              loop. The next-node for a continue is the top of this loop.
              */
              let node = self.live_node(expr.id, expr.span);
              self.with_loop_nodes(blk.id, succ, node, |this| {

                 // the construction of a closure itself is not important,
                 // but we have to consider the closed over variables.
                 let caps = match this.ir.capture_info_map.find(&expr.id) {
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
                    self.propagate_through_expr(arm.body, succ);
                let guard_succ =
                    self.propagate_through_opt_expr(arm.guard, body_succ);
                let arm_succ =
                    self.define_bindings_in_arm_pats(arm.pats.as_slice(),
                                                     guard_succ);
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

              match self.break_ln.find(&sc) {
                  Some(&b) => b,
                  None => self.ir.tcx.sess.span_bug(expr.span,
                                                    "break to unknown label")
              }
          }

          ExprAgain(opt_label) => {
              // Find which label this expr continues to
              let sc = self.find_loop_scope(opt_label, expr.id, expr.span);

              // Now that we know the label we're going to,
              // look it up in the continue loop nodes table

              match self.cont_ln.find(&sc) {
                  Some(&b) => b,
                  None => self.ir.tcx.sess.span_bug(expr.span,
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

          ExprAssignOp(_, l, r) => {
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

          ExprVec(ref exprs) => {
            self.propagate_through_exprs(exprs.as_slice(), succ)
          }

          ExprRepeat(element, count) => {
            let succ = self.propagate_through_expr(count, succ);
            self.propagate_through_expr(element, succ)
          }

          ExprStruct(_, ref fields, with_expr) => {
            let succ = self.propagate_through_opt_expr(with_expr, succ);
            fields.iter().rev().fold(succ, |succ, field| {
                self.propagate_through_expr(field.expr, succ)
            })
          }

          ExprCall(f, ref args) => {
            // calling a fn with bot return type means that the fn
            // will fail, and hence the successors can be ignored
            let t_ret = ty::ty_fn_ret(ty::expr_ty(self.ir.tcx, f));
            let succ = if ty::type_is_bot(t_ret) {self.s.exit_ln}
                       else {succ};
            let succ = self.propagate_through_exprs(args.as_slice(), succ);
            self.propagate_through_expr(f, succ)
          }

          ExprMethodCall(_, _, ref args) => {
            // calling a method with bot return type means that the method
            // will fail, and hence the successors can be ignored
            let t_ret = ty::node_id_to_type(self.ir.tcx, expr.id);
            let succ = if ty::type_is_bot(t_ret) {self.s.exit_ln}
                       else {succ};
            self.propagate_through_exprs(args.as_slice(), succ)
          }

          ExprTup(ref exprs) => {
            self.propagate_through_exprs(exprs.as_slice(), succ)
          }

          ExprBinary(op, l, r) if ast_util::lazy_binop(op) => {
            let r_succ = self.propagate_through_expr(r, succ);

            let ln = self.live_node(expr.id, expr.span);
            self.init_from_succ(ln, succ);
            self.merge_from_succ(ln, r_succ, false);

            self.propagate_through_expr(l, ln)
          }

          ExprIndex(l, r) |
          ExprBinary(_, l, r) |
          ExprBox(l, r) => {
            self.propagate_through_exprs([l, r], succ)
          }

          ExprAddrOf(_, e) |
          ExprCast(e, _) |
          ExprUnary(_, e) |
          ExprParen(e) => {
            self.propagate_through_expr(e, succ)
          }

          ExprInlineAsm(ref ia) => {
            let succ = ia.outputs.iter().rev().fold(succ, |succ, &(_, expr)| {
                // see comment on lvalues in
                // propagate_through_lvalue_components()
                let succ = self.write_lvalue(expr, succ, ACC_WRITE);
                self.propagate_through_lvalue_components(expr, succ)
            });
            // Inputs are executed first. Propagate last because of rev order
            ia.inputs.iter().rev().fold(succ, |succ, &(_, expr)| {
                self.propagate_through_expr(expr, succ)
            })
          }

          ExprLit(..) => {
            succ
          }

          ExprBlock(blk) => {
            self.propagate_through_block(blk, succ)
          }

          ExprMac(..) => {
            self.ir.tcx.sess.span_bug(expr.span, "unexpanded macro");
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
            ExprPath(_) => succ,
            ExprField(e, _, _) => self.propagate_through_expr(e, succ),
            _ => self.propagate_through_expr(expr, succ)
        }
    }

    // see comment on propagate_through_lvalue()
    fn write_lvalue(&mut self, expr: &Expr, succ: LiveNode, acc: uint)
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

    fn access_path(&mut self, expr: &Expr, succ: LiveNode, acc: uint)
                   -> LiveNode {
        let def = self.ir.tcx.def_map.borrow().get_copy(&expr.id);
        match moved_variable_node_id_from_def(def) {
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

    fn propagate_through_loop(&mut self,
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
               expr.id, block_to_str(body));

        let cond_ln = self.propagate_through_opt_expr(cond, ln);
        let body_ln = self.with_loop_nodes(expr.id, succ, ln, |this| {
            this.propagate_through_block(body, cond_ln)
        });

        // repeat until fixed point is reached:
        while self.merge_from_succ(ln, body_ln, first_merge) {
            first_merge = false;
            assert!(cond_ln == self.propagate_through_opt_expr(cond,
                                                                    ln));
            assert!(body_ln == self.with_loop_nodes(expr.id, succ, ln,
            |this| this.propagate_through_block(body, cond_ln)));
        }

        cond_ln
    }

    fn with_loop_nodes<R>(&mut self,
                          loop_node_id: NodeId,
                          break_ln: LiveNode,
                          cont_ln: LiveNode,
                          f: |&mut Liveness<'a>| -> R)
                          -> R {
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

fn check_local(this: &mut Liveness, local: &Local) {
    match local.init {
        Some(_) => {
            this.warn_about_unused_or_dead_vars_in_pat(local.pat);
        },
        None => {
            this.pat_bindings(local.pat, |this, ln, var, sp, id| {
                this.warn_about_unused(sp, id, ln, var);
            })
        }
    }

    visit::walk_local(this, local, ());
}

fn check_arm(this: &mut Liveness, arm: &Arm) {
    this.arm_pats_bindings(arm.pats.as_slice(), |this, ln, var, sp, id| {
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

      ExprAssignOp(_, l, _) => {
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
      ExprVstore(..) | ExprVec(..) | ExprTup(..) |
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

impl<'a> Liveness<'a> {
    fn check_ret(&self,
                 id: NodeId,
                 sp: Span,
                 _fk: &FnKind,
                 entry_ln: LiveNode,
                 body: &Block) {
        if self.live_on_entry(entry_ln, self.s.no_ret_var).is_some() {
            // if no_ret_var is live, then we fall off the end of the
            // function without any kind of return expression:

            let t_ret = ty::ty_fn_ret(ty::node_id_to_type(self.ir.tcx, id));
            if ty::type_is_nil(t_ret) {
                // for nil return types, it is ok to not return a value expl.
            } else if ty::type_is_bot(t_ret) {
                // for bot return types, not ok.  Function should fail.
                self.ir.tcx.sess.span_err(
                    sp, "some control paths may return");
            } else {
                let ends_with_stmt = match body.expr {
                    None if body.stmts.len() > 0 =>
                        match body.stmts.last().unwrap().node {
                            StmtSemi(e, _) => {
                                let t_stmt = ty::expr_ty(self.ir.tcx, e);
                                ty::get(t_stmt).sty == ty::get(t_ret).sty
                            },
                            _ => false
                        },
                    _ => false
                };
                if ends_with_stmt {
                    let last_stmt = body.stmts.last().unwrap();
                    let original_span = original_sp(last_stmt.span, sp);
                    let span_semicolon = Span {
                        lo: original_span.hi - BytePos(1),
                        hi: original_span.hi,
                        expn_info: original_span.expn_info
                    };
                    self.ir.tcx.sess.span_note(
                        span_semicolon, "consider removing this semicolon:");
                }
                self.ir.tcx.sess.span_err(
                    sp, "not all control paths return a value");
           }
        }
    }

    fn check_lvalue(&mut self, expr: &Expr) {
        match expr.node {
          ExprPath(_) => {
            match self.ir.tcx.def_map.borrow().get_copy(&expr.id) {
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
                match moved_variable_node_id_from_def(def) {
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

    fn should_warn(&self, var: Variable) -> Option<StrBuf> {
        let name = self.ir.variable_name(var);
        if name.len() == 0 || name.as_slice()[0] == ('_' as u8) {
            None
        } else {
            Some(name)
        }
    }

    fn warn_about_unused_args(&self, decl: &FnDecl, entry_ln: LiveNode) {
        for arg in decl.inputs.iter() {
            pat_util::pat_bindings(&self.ir.tcx.def_map,
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

    fn warn_about_unused_or_dead_vars_in_pat(&mut self, pat: &Pat) {
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
                    self.ir.tcx.sess.add_lint(UnusedVariable, id, sp,
                        format_strbuf!("variable `{}` is assigned to, \
                                        but never used",
                                       *name));
                } else {
                    self.ir.tcx.sess.add_lint(UnusedVariable, id, sp,
                        format_strbuf!("unused variable: `{}`", *name));
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
            let r = self.should_warn(var);
            for name in r.iter() {
                self.ir.tcx.sess.add_lint(DeadAssignment, id, sp,
                    format_strbuf!("value assigned to `{}` is never read",
                                   *name));
            }
        }
    }
 }
