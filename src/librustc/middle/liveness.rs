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
 * ## The `users` struct
 *
 * At each live node `N`, we track three pieces of information for each
 * variable `V` (these are encapsulated in the `users` struct):
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

use dvec::DVec;
use std::map::HashMap;
use syntax::{visit, ast_util};
use syntax::print::pprust::{expr_to_str, block_to_str};
use visit::vt;
use syntax::codemap::span;
use syntax::ast::*;
use io::WriterUtil;
use capture::{cap_move, cap_drop, cap_copy, cap_ref};

export check_crate;
export last_use_map;

// Maps from an expr id to a list of variable ids for which this expr
// is the last use.  Typically, the expr is a path and the node id is
// the local/argument/etc that the path refers to.  However, it also
// possible for the expr to be a closure, in which case the list is a
// list of closed over variables that can be moved into the closure.
//
// Very subtle (#2633): borrowck will remove entries from this table
// if it detects an outstanding loan (that is, the addr is taken).
type last_use_map = HashMap<node_id, @DVec<node_id>>;

enum Variable = uint;
enum LiveNode = uint;

impl Variable : cmp::Eq {
    pure fn eq(other: &Variable) -> bool { *self == *(*other) }
    pure fn ne(other: &Variable) -> bool { *self != *(*other) }
}

impl LiveNode : cmp::Eq {
    pure fn eq(other: &LiveNode) -> bool { *self == *(*other) }
    pure fn ne(other: &LiveNode) -> bool { *self != *(*other) }
}

enum LiveNodeKind {
    FreeVarNode(span),
    ExprNode(span),
    VarDefNode(span),
    ExitNode
}

impl LiveNodeKind : cmp::Eq {
    pure fn eq(other: &LiveNodeKind) -> bool {
        match self {
            FreeVarNode(e0a) => {
                match (*other) {
                    FreeVarNode(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ExprNode(e0a) => {
                match (*other) {
                    ExprNode(e0b) => e0a == e0b,
                    _ => false
                }
            }
            VarDefNode(e0a) => {
                match (*other) {
                    VarDefNode(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ExitNode => {
                match (*other) {
                    ExitNode => true,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &LiveNodeKind) -> bool { !self.eq(other) }
}

fn live_node_kind_to_str(lnk: LiveNodeKind, cx: ty::ctxt) -> ~str {
    let cm = cx.sess.codemap;
    match lnk {
        FreeVarNode(s) => fmt!("Free var node [%s]", cm.span_to_str(s)),
        ExprNode(s)    => fmt!("Expr node [%s]", cm.span_to_str(s)),
        VarDefNode(s)  => fmt!("Var def node [%s]", cm.span_to_str(s)),
        ExitNode       => ~"Exit node"
    }
}

fn check_crate(tcx: ty::ctxt,
               method_map: typeck::method_map,
               crate: @crate) -> last_use_map {
    let visitor = visit::mk_vt(@{
        visit_fn: visit_fn,
        visit_local: visit_local,
        visit_expr: visit_expr,
        visit_arm: visit_arm,
        .. *visit::default_visitor()
    });

    let last_use_map = HashMap();
    let initial_maps = @IrMaps(tcx, method_map, last_use_map);
    visit::visit_crate(*crate, initial_maps, visitor);
    tcx.sess.abort_if_errors();
    return last_use_map;
}

impl LiveNode: to_str::ToStr {
    pure fn to_str() -> ~str { fmt!("ln(%u)", *self) }
}

impl Variable: to_str::ToStr {
    pure fn to_str() -> ~str { fmt!("v(%u)", *self) }
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
    pure fn is_valid() -> bool { *self != uint::max_value }
}

fn invalid_node() -> LiveNode { LiveNode(uint::max_value) }

struct CaptureInfo {
    ln: LiveNode,
    is_move: bool,
    var_nid: node_id
}

enum LocalKind {
    FromMatch(binding_mode),
    FromLetWithInitializer,
    FromLetNoInitializer
}

struct LocalInfo {
    id: node_id,
    ident: ident,
    is_mutbl: bool,
    kind: LocalKind,
}

enum VarKind {
    Arg(node_id, ident, rmode),
    Local(LocalInfo),
    Self,
    ImplicitRet
}

fn relevant_def(def: def) -> Option<node_id> {
    match def {
      def_binding(nid, _) |
      def_arg(nid, _) |
      def_local(nid, _) => Some(nid),

      _ => None
    }
}

struct IrMaps {
    tcx: ty::ctxt,
    method_map: typeck::method_map,
    last_use_map: last_use_map,

    mut num_live_nodes: uint,
    mut num_vars: uint,
    live_node_map: HashMap<node_id, LiveNode>,
    variable_map: HashMap<node_id, Variable>,
    capture_map: HashMap<node_id, @~[CaptureInfo]>,
    mut var_kinds: ~[VarKind],
    mut lnks: ~[LiveNodeKind],
}

fn IrMaps(tcx: ty::ctxt, method_map: typeck::method_map,
          last_use_map: last_use_map) -> IrMaps {
    IrMaps {
        tcx: tcx,
        method_map: method_map,
        last_use_map: last_use_map,
        num_live_nodes: 0,
        num_vars: 0,
        live_node_map: HashMap(),
        variable_map: HashMap(),
        capture_map: HashMap(),
        var_kinds: ~[],
        lnks: ~[]
    }
}

impl IrMaps {
    fn add_live_node(lnk: LiveNodeKind) -> LiveNode {
        let ln = LiveNode(self.num_live_nodes);
        self.lnks.push(lnk);
        self.num_live_nodes += 1;

        debug!("%s is of kind %s", ln.to_str(),
               live_node_kind_to_str(lnk, self.tcx));

        ln
    }

    fn add_live_node_for_node(node_id: node_id, lnk: LiveNodeKind) {
        let ln = self.add_live_node(lnk);
        self.live_node_map.insert(node_id, ln);

        debug!("%s is node %d", ln.to_str(), node_id);
    }

    fn add_variable(vk: VarKind) -> Variable {
        let v = Variable(self.num_vars);
        self.var_kinds.push(vk);
        self.num_vars += 1;

        match vk {
            Local(LocalInfo {id:node_id, _}) |
            Arg(node_id, _, _) => {
                self.variable_map.insert(node_id, v);
            }
            Self | ImplicitRet => {
            }
        }

        debug!("%s is %?", v.to_str(), vk);

        v
    }

    fn variable(node_id: node_id, span: span) -> Variable {
        match self.variable_map.find(node_id) {
          Some(var) => var,
          None => {
            self.tcx.sess.span_bug(
                span, fmt!("No variable registered for id %d", node_id));
          }
        }
    }

    fn variable_name(var: Variable) -> ~str {
        match copy self.var_kinds[*var] {
            Local(LocalInfo {ident: nm, _}) |
            Arg(_, nm, _) => self.tcx.sess.str_of(nm),
            Self => ~"self",
            ImplicitRet => ~"<implicit-ret>"
        }
    }

    fn set_captures(node_id: node_id, +cs: ~[CaptureInfo]) {
        self.capture_map.insert(node_id, @cs);
    }

    fn captures(expr: @expr) -> @~[CaptureInfo] {
        match self.capture_map.find(expr.id) {
          Some(caps) => caps,
          None => {
            self.tcx.sess.span_bug(expr.span, ~"no registered caps");
          }
        }
    }

    fn lnk(ln: LiveNode) -> LiveNodeKind {
        self.lnks[*ln]
    }

    fn add_last_use(expr_id: node_id, var: Variable) {
        let vk = self.var_kinds[*var];
        debug!("Node %d is a last use of variable %?", expr_id, vk);
        match vk {
          Arg(id, _, by_move) |
          Arg(id, _, by_copy) |
          Local(LocalInfo {id: id, kind: FromLetNoInitializer, _}) |
          Local(LocalInfo {id: id, kind: FromLetWithInitializer, _}) |
          Local(LocalInfo {id: id, kind: FromMatch(bind_by_value), _}) |
          Local(LocalInfo {id: id, kind: FromMatch(bind_by_ref(_)), _}) |
          Local(LocalInfo {id: id, kind: FromMatch(bind_by_move), _}) => {
            let v = match self.last_use_map.find(expr_id) {
              Some(v) => v,
              None => {
                let v = @DVec();
                self.last_use_map.insert(expr_id, v);
                v
              }
            };

            (*v).push(id);
          }
          Arg(_, _, by_ref) |
          Arg(_, _, by_val) | Self | ImplicitRet |
          Local(LocalInfo {kind: FromMatch(bind_by_implicit_ref), _}) => {
            debug!("--but it is not owned");
          }
        }
    }
}

fn visit_fn(fk: visit::fn_kind, decl: fn_decl, body: blk,
            sp: span, id: node_id, &&self: @IrMaps, v: vt<@IrMaps>) {
    debug!("visit_fn: id=%d", id);
    let _i = util::common::indenter();

    // swap in a new set of IR maps for this function body:
    let fn_maps = @IrMaps(self.tcx, self.method_map,
                          self.last_use_map);

    debug!("creating fn_maps: %x", ptr::addr_of(&(*fn_maps)) as uint);

    for decl.inputs.each |arg| {
        let mode = ty::resolved_mode(self.tcx, arg.mode);
        do pat_util::pat_bindings(self.tcx.def_map, arg.pat)
                |_bm, arg_id, _x, path| {
            debug!("adding argument %d", arg_id);
            let ident = ast_util::path_to_ident(path);
            (*fn_maps).add_variable(Arg(arg_id, ident, mode));
        }
    };

    // gather up the various local variables, significant expressions,
    // and so forth:
    visit::visit_fn(fk, decl, body, sp, id, fn_maps, v);

    // Special nodes and variables:
    // - exit_ln represents the end of the fn, either by return or fail
    // - implicit_ret_var is a pseudo-variable that represents
    //   an implicit return
    let specials = {
        exit_ln: (*fn_maps).add_live_node(ExitNode),
        fallthrough_ln: (*fn_maps).add_live_node(ExitNode),
        no_ret_var: (*fn_maps).add_variable(ImplicitRet)
    };

    // compute liveness
    let lsets = @Liveness(fn_maps, specials);
    let entry_ln = (*lsets).compute(decl, body);

    // check for various error conditions
    let check_vt = visit::mk_vt(@{
        visit_fn: check_fn,
        visit_local: check_local,
        visit_expr: check_expr,
        visit_arm: check_arm,
        .. *visit::default_visitor()
    });
    check_vt.visit_block(body, lsets, check_vt);
    lsets.check_ret(id, sp, fk, entry_ln);
    lsets.warn_about_unused_args(decl, entry_ln);
}

fn visit_local(local: @local, &&self: @IrMaps, vt: vt<@IrMaps>) {
    let def_map = self.tcx.def_map;
    do pat_util::pat_bindings(def_map, local.node.pat) |_bm, p_id, sp, path| {
        debug!("adding local variable %d", p_id);
        let name = ast_util::path_to_ident(path);
        self.add_live_node_for_node(p_id, VarDefNode(sp));
        let kind = match local.node.init {
          Some(_) => FromLetWithInitializer,
          None => FromLetNoInitializer
        };
        self.add_variable(Local(LocalInfo {
          id: p_id,
          ident: name,
          is_mutbl: local.node.is_mutbl,
          kind: kind
        }));
    }
    visit::visit_local(local, self, vt);
}

fn visit_arm(arm: arm, &&self: @IrMaps, vt: vt<@IrMaps>) {
    let def_map = self.tcx.def_map;
    for arm.pats.each |pat| {
        do pat_util::pat_bindings(def_map, *pat) |bm, p_id, sp, path| {
            debug!("adding local variable %d from match with bm %?",
                   p_id, bm);
            let name = ast_util::path_to_ident(path);
            self.add_live_node_for_node(p_id, VarDefNode(sp));
            self.add_variable(Local(LocalInfo {
                id: p_id,
                ident: name,
                is_mutbl: false,
                kind: FromMatch(bm)
            }));
        }
    }
    visit::visit_arm(arm, self, vt);
}

fn visit_expr(expr: @expr, &&self: @IrMaps, vt: vt<@IrMaps>) {
    match expr.node {
      // live nodes required for uses or definitions of variables:
      expr_path(_) => {
        let def = self.tcx.def_map.get(expr.id);
        debug!("expr %d: path that leads to %?", expr.id, def);
        if relevant_def(def).is_some() {
            self.add_live_node_for_node(expr.id, ExprNode(expr.span));
        }
        visit::visit_expr(expr, self, vt);
      }
      expr_fn(_, _, _, cap_clause) |
      expr_fn_block(_, _, cap_clause) => {
        // Interesting control flow (for loops can contain labeled
        // breaks or continues)
        self.add_live_node_for_node(expr.id, ExprNode(expr.span));

        // Make a live_node for each captured variable, with the span
        // being the location that the variable is used.  This results
        // in better error messages than just pointing at the closure
        // construction site.
        let proto = ty::ty_fn_proto(ty::expr_ty(self.tcx, expr));
        let cvs = capture::compute_capture_vars(self.tcx, expr.id,
                                                proto, cap_clause);
        let mut call_caps = ~[];
        for cvs.each |cv| {
            match relevant_def(cv.def) {
              Some(rv) => {
                let cv_ln = self.add_live_node(FreeVarNode(cv.span));
                let is_move = match cv.mode {
                  cap_move | cap_drop => true, // var must be dead afterwards
                  cap_copy | cap_ref => false // var can still be used
                };
                call_caps.push(CaptureInfo {ln: cv_ln,
                                            is_move: is_move,
                                            var_nid: rv});
              }
              None => {}
            }
        }
        self.set_captures(expr.id, call_caps);

        visit::visit_expr(expr, self, vt);
      }

      // live nodes required for interesting control flow:
      expr_if(*) | expr_match(*) | expr_while(*) | expr_loop(*) => {
        self.add_live_node_for_node(expr.id, ExprNode(expr.span));
        visit::visit_expr(expr, self, vt);
      }
      expr_binary(op, _, _) if ast_util::lazy_binop(op) => {
        self.add_live_node_for_node(expr.id, ExprNode(expr.span));
        visit::visit_expr(expr, self, vt);
      }

      // otherwise, live nodes are not required:
      expr_index(*) | expr_field(*) | expr_vstore(*) |
      expr_vec(*) | expr_rec(*) | expr_call(*) | expr_tup(*) |
      expr_log(*) | expr_binary(*) |
      expr_assert(*) | expr_addr_of(*) | expr_copy(*) |
      expr_loop_body(*) | expr_do_body(*) | expr_cast(*) |
      expr_unary(*) | expr_fail(*) |
      expr_break(_) | expr_again(_) | expr_lit(_) | expr_ret(*) |
      expr_block(*) | expr_unary_move(*) | expr_assign(*) |
      expr_swap(*) | expr_assign_op(*) | expr_mac(*) | expr_struct(*) |
      expr_repeat(*) | expr_paren(*) => {
          visit::visit_expr(expr, self, vt);
      }
    }
}

// ______________________________________________________________________
// Computing liveness sets
//
// Actually we compute just a bit more than just liveness, but we use
// the same basic propagation framework in all cases.

type users = {
    reader: LiveNode,
    writer: LiveNode,
    used: bool
};

fn invalid_users() -> users {
    {reader: invalid_node(), writer: invalid_node(), used: false}
}

type Specials = {
    exit_ln: LiveNode,
    fallthrough_ln: LiveNode,
    no_ret_var: Variable
};

const ACC_READ: uint = 1u;
const ACC_WRITE: uint = 2u;
const ACC_USE: uint = 4u;

type LiveNodeMap = HashMap<node_id, LiveNode>;

struct Liveness {
    tcx: ty::ctxt,
    ir: @IrMaps,
    s: Specials,
    successors: ~[mut LiveNode],
    users: ~[mut users],
    // The list of node IDs for the nested loop scopes
    // we're in.
    loop_scope: DVec<node_id>,
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
        successors:
            vec::to_mut(
                vec::from_elem(ir.num_live_nodes,
                               invalid_node())),
        users:
            vec::to_mut(
                vec::from_elem(ir.num_live_nodes * ir.num_vars,
                               invalid_users())),
        loop_scope: DVec(),
        break_ln: HashMap(),
        cont_ln: HashMap()
    }
}

impl Liveness {
    fn live_node(node_id: node_id, span: span) -> LiveNode {
        match self.ir.live_node_map.find(node_id) {
          Some(ln) => ln,
          None => {
            // This must be a mismatch between the ir_map construction
            // above and the propagation code below; the two sets of
            // code have to agree about which AST nodes are worth
            // creating liveness nodes for.
            self.tcx.sess.span_bug(
                span, fmt!("No live node registered for node %d",
                           node_id));
          }
        }
    }

    fn variable_from_path(expr: @expr) -> Option<Variable> {
        match expr.node {
          expr_path(_) => {
            let def = self.tcx.def_map.get(expr.id);
            relevant_def(def).map(
                |rdef| self.variable(*rdef, expr.span)
            )
          }
          _ => None
        }
    }

    fn variable(node_id: node_id, span: span) -> Variable {
        (*self.ir).variable(node_id, span)
    }

    fn variable_from_def_map(node_id: node_id,
                             span: span) -> Option<Variable> {
        match self.tcx.def_map.find(node_id) {
          Some(def) => {
            relevant_def(def).map(
                |rdef| self.variable(*rdef, span)
            )
          }
          None => {
            self.tcx.sess.span_bug(
                span, ~"Not present in def map")
          }
        }
    }

    fn pat_bindings(pat: @pat, f: fn(LiveNode, Variable, span)) {
        let def_map = self.tcx.def_map;
        do pat_util::pat_bindings(def_map, pat) |_bm, p_id, sp, _n| {
            let ln = self.live_node(p_id, sp);
            let var = self.variable(p_id, sp);
            f(ln, var, sp);
        }
    }

    fn arm_pats_bindings(pats: &[@pat], f: fn(LiveNode, Variable, span)) {
        // only consider the first pattern; any later patterns must have
        // the same bindings, and we also consider the first pattern to be
        // the "authoratative" set of ids
        if !pats.is_empty() {
            self.pat_bindings(pats[0], f)
        }
    }

    fn define_bindings_in_pat(pat: @pat, succ: LiveNode) -> LiveNode {
        self.define_bindings_in_arm_pats([pat], succ)
    }

    fn define_bindings_in_arm_pats(pats: &[@pat],
                                   succ: LiveNode) -> LiveNode {
        let mut succ = succ;
        do self.arm_pats_bindings(pats) |ln, var, _sp| {
            self.init_from_succ(ln, succ);
            self.define(ln, var);
            succ = ln;
        }
        succ
    }

    fn idx(ln: LiveNode, var: Variable) -> uint {
        *ln * self.ir.num_vars + *var
    }

    fn live_on_entry(ln: LiveNode, var: Variable)
        -> Option<LiveNodeKind> {

        assert ln.is_valid();
        let reader = self.users[self.idx(ln, var)].reader;
        if reader.is_valid() {Some((*self.ir).lnk(reader))} else {None}
    }

    /*
    Is this variable live on entry to any of its successor nodes?
    */
    fn live_on_exit(ln: LiveNode, var: Variable)
        -> Option<LiveNodeKind> {

        self.live_on_entry(copy self.successors[*ln], var)
    }

    fn used_on_entry(ln: LiveNode, var: Variable) -> bool {
        assert ln.is_valid();
        self.users[self.idx(ln, var)].used
    }

    fn assigned_on_entry(ln: LiveNode, var: Variable)
        -> Option<LiveNodeKind> {

        assert ln.is_valid();
        let writer = self.users[self.idx(ln, var)].writer;
        if writer.is_valid() {Some((*self.ir).lnk(writer))} else {None}
    }

    fn assigned_on_exit(ln: LiveNode, var: Variable)
        -> Option<LiveNodeKind> {

        self.assigned_on_entry(copy self.successors[*ln], var)
    }

    fn indices(ln: LiveNode, op: fn(uint)) {
        let node_base_idx = self.idx(ln, Variable(0));
        for uint::range(0, self.ir.num_vars) |var_idx| {
            op(node_base_idx + var_idx)
        }
    }

    fn indices2(ln: LiveNode, succ_ln: LiveNode,
                op: fn(uint, uint)) {
        let node_base_idx = self.idx(ln, Variable(0u));
        let succ_base_idx = self.idx(succ_ln, Variable(0u));
        for uint::range(0u, self.ir.num_vars) |var_idx| {
            op(node_base_idx + var_idx, succ_base_idx + var_idx);
        }
    }

    fn write_vars(wr: io::Writer,
                  ln: LiveNode,
                  test: fn(uint) -> LiveNode) {
        let node_base_idx = self.idx(ln, Variable(0));
        for uint::range(0, self.ir.num_vars) |var_idx| {
            let idx = node_base_idx + var_idx;
            if test(idx).is_valid() {
                wr.write_str(~" ");
                wr.write_str(Variable(var_idx).to_str());
            }
        }
    }

    fn find_loop_scope(opt_label: Option<ident>, id: node_id, sp: span)
        -> node_id {
        match opt_label {
            Some(_) => // Refers to a labeled loop. Use the results of resolve
                      // to find with one
                match self.tcx.def_map.find(id) {
                    Some(def_label(loop_id)) => loop_id,
                    _ => self.tcx.sess.span_bug(sp, ~"Label on break/loop \
                                                    doesn't refer to a loop")
                },
            None =>
                // Vanilla 'break' or 'loop', so use the enclosing
                // loop scope
                if self.loop_scope.len() == 0 {
                    self.tcx.sess.span_bug(sp, ~"break outside loop");
                }
                else {
                    self.loop_scope.last()
                }
        }
    }

    fn ln_str(ln: LiveNode) -> ~str {
        do io::with_str_writer |wr| {
            wr.write_str(~"[ln(");
            wr.write_uint(*ln);
            wr.write_str(~") of kind ");
            wr.write_str(fmt!("%?", copy self.ir.lnks[*ln]));
            wr.write_str(~" reads");
            self.write_vars(wr, ln, |idx| self.users[idx].reader );
            wr.write_str(~"  writes");
            self.write_vars(wr, ln, |idx| self.users[idx].writer );
            wr.write_str(~" ");
            wr.write_str(~" precedes ");
            wr.write_str((copy self.successors[*ln]).to_str());
            wr.write_str(~"]");
        }
    }

    fn init_empty(ln: LiveNode, succ_ln: LiveNode) {
        self.successors[*ln] = succ_ln;

        // It is not necessary to initialize the
        // values to empty because this is the value
        // they have when they are created, and the sets
        // only grow during iterations.
        //
        // self.indices(ln) { |idx|
        //     self.users[idx] = invalid_users();
        // }
    }

    fn init_from_succ(ln: LiveNode, succ_ln: LiveNode) {
        // more efficient version of init_empty() / merge_from_succ()
        self.successors[*ln] = succ_ln;
        self.indices2(ln, succ_ln, |idx, succ_idx| {
            self.users[idx] = self.users[succ_idx]
        });
        debug!("init_from_succ(ln=%s, succ=%s)",
               self.ln_str(ln), self.ln_str(succ_ln));
    }

    fn merge_from_succ(ln: LiveNode, succ_ln: LiveNode,
                       first_merge: bool) -> bool {
        if ln == succ_ln { return false; }

        let mut changed = false;
        do self.indices2(ln, succ_ln) |idx, succ_idx| {
            changed |= copy_if_invalid(copy self.users[succ_idx].reader,
                                       &mut self.users[idx].reader);
            changed |= copy_if_invalid(copy self.users[succ_idx].writer,
                                       &mut self.users[idx].writer);
            if self.users[succ_idx].used && !self.users[idx].used {
                self.users[idx].used = true;
                changed = true;
            }
        }

        debug!("merge_from_succ(ln=%s, succ=%s, first_merge=%b, changed=%b)",
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
    fn define(writer: LiveNode, var: Variable) {
        let idx = self.idx(writer, var);
        self.users[idx].reader = invalid_node();
        self.users[idx].writer = invalid_node();

        debug!("%s defines %s (idx=%u): %s", writer.to_str(), var.to_str(),
               idx, self.ln_str(writer));
    }

    // Either read, write, or both depending on the acc bitset
    fn acc(ln: LiveNode, var: Variable, acc: uint) {
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
            self.users[idx].used = true;
        }

        debug!("%s accesses[%x] %s: %s",
               ln.to_str(), acc, var.to_str(), self.ln_str(ln));
    }

    // _______________________________________________________________________

    fn compute(decl: fn_decl, body: blk) -> LiveNode {
        // if there is a `break` or `again` at the top level, then it's
        // effectively a return---this only occurs in `for` loops,
        // where the body is really a closure.

        debug!("compute: using id for block, %s", block_to_str(body,
                      self.tcx.sess.intr()));

        let entry_ln: LiveNode =
            self.with_loop_nodes(body.node.id, self.s.exit_ln, self.s.exit_ln,
              || { self.propagate_through_fn_block(decl, body) });

        // hack to skip the loop unless debug! is enabled:
        debug!("^^ liveness computation results for body %d (entry=%s)",
               {
                   for uint::range(0u, self.ir.num_live_nodes) |ln_idx| {
                       debug!("%s", self.ln_str(LiveNode(ln_idx)));
                   }
                   body.node.id
               },
               entry_ln.to_str());

        entry_ln
    }

    fn propagate_through_fn_block(decl: fn_decl, blk: blk) -> LiveNode {
        // inputs passed by & mode should be considered live on exit:
        for decl.inputs.each |arg| {
            match ty::resolved_mode(self.tcx, arg.mode) {
              by_ref | by_val => {
                // These are "non-owned" modes, so register a read at
                // the end.  This will prevent us from moving out of
                // such variables but also prevent us from registering
                // last uses and so forth.
                do pat_util::pat_bindings(self.tcx.def_map, arg.pat)
                        |_bm, arg_id, _sp, _path| {
                    let var = self.variable(arg_id, blk.span);
                    self.acc(self.s.exit_ln, var, ACC_READ);
                }
              }
              by_move | by_copy => {
                // These are owned modes.  If we don't use the
                // variable, nobody will.
              }
            }
        }

        // the fallthrough exit is only for those cases where we do not
        // explicitly return:
        self.init_from_succ(self.s.fallthrough_ln, self.s.exit_ln);
        if blk.node.expr.is_none() {
            self.acc(self.s.fallthrough_ln, self.s.no_ret_var, ACC_READ)
        }

        self.propagate_through_block(blk, self.s.fallthrough_ln)
    }

    fn propagate_through_block(blk: blk, succ: LiveNode) -> LiveNode {
        let succ = self.propagate_through_opt_expr(blk.node.expr, succ);
        do blk.node.stmts.foldr(succ) |stmt, succ| {
            self.propagate_through_stmt(*stmt, succ)
        }
    }

    fn propagate_through_stmt(stmt: @stmt, succ: LiveNode) -> LiveNode {
        match stmt.node {
          stmt_decl(decl, _) => {
            return self.propagate_through_decl(decl, succ);
          }

          stmt_expr(expr, _) | stmt_semi(expr, _) => {
            return self.propagate_through_expr(expr, succ);
          }
        }
    }

    fn propagate_through_decl(decl: @decl, succ: LiveNode) -> LiveNode {
        match decl.node {
          decl_local(locals) => {
            do locals.foldr(succ) |local, succ| {
                self.propagate_through_local(*local, succ)
            }
          }
          decl_item(_) => {
            succ
          }
        }
    }

    fn propagate_through_local(local: @local, succ: LiveNode) -> LiveNode {
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

        let succ = self.propagate_through_opt_expr(local.node.init, succ);
        self.define_bindings_in_pat(local.node.pat, succ)
    }

    fn propagate_through_exprs(exprs: ~[@expr],
                               succ: LiveNode) -> LiveNode {
        do exprs.foldr(succ) |expr, succ| {
            self.propagate_through_expr(*expr, succ)
        }
    }

    fn propagate_through_opt_expr(opt_expr: Option<@expr>,
                                  succ: LiveNode) -> LiveNode {
        do opt_expr.foldl(succ) |succ, expr| {
            self.propagate_through_expr(*expr, *succ)
        }
    }

    fn propagate_through_expr(expr: @expr, succ: LiveNode) -> LiveNode {
        debug!("propagate_through_expr: %s",
             expr_to_str(expr, self.tcx.sess.intr()));

        match expr.node {
          // Interesting cases with control flow or which gen/kill

          expr_path(_) => {
              self.access_path(expr, succ, ACC_READ | ACC_USE)
          }

          expr_field(e, _, _) => {
              self.propagate_through_expr(e, succ)
          }

          expr_fn(_, _, blk, _) | expr_fn_block(_, blk, _) => {
              debug!("%s is an expr_fn or expr_fn_block",
                   expr_to_str(expr, self.tcx.sess.intr()));

              /*
              The next-node for a break is the successor of the entire
              loop. The next-node for a continue is the top of this loop.
              */
              self.with_loop_nodes(blk.node.id, succ,
                  self.live_node(expr.id, expr.span), || {

                 // the construction of a closure itself is not important,
                 // but we have to consider the closed over variables.
                 let caps = (*self.ir).captures(expr);
                 do (*caps).foldr(succ) |cap, succ| {
                     self.init_from_succ(cap.ln, succ);
                     let var = self.variable(cap.var_nid, expr.span);
                     self.acc(cap.ln, var, ACC_READ | ACC_USE);
                     cap.ln
                 }
              })
          }

          expr_if(cond, then, els) => {
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

          expr_while(cond, blk) => {
            self.propagate_through_loop(expr, Some(cond), blk, succ)
          }

          // Note that labels have been resolved, so we don't need to look
          // at the label ident
          expr_loop(blk, _) => {
            self.propagate_through_loop(expr, None, blk, succ)
          }

          expr_match(e, arms) => {
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
            for arms.each |arm| {
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

          expr_ret(o_e) | expr_fail(o_e) => {
            // ignore succ and subst exit_ln:
            self.propagate_through_opt_expr(o_e, self.s.exit_ln)
          }

          expr_break(opt_label) => {
              // Find which label this break jumps to
              let sc = self.find_loop_scope(opt_label, expr.id, expr.span);

              // Now that we know the label we're going to,
              // look it up in the break loop nodes table

              match self.break_ln.find(sc) {
                  Some(b) => b,
                  None => self.tcx.sess.span_bug(expr.span,
                                ~"Break to unknown label")
              }
          }

          expr_again(opt_label) => {
              // Find which label this expr continues to to
              let sc = self.find_loop_scope(opt_label, expr.id, expr.span);

              // Now that we know the label we're going to,
              // look it up in the continue loop nodes table

              match self.cont_ln.find(sc) {
                  Some(b) => b,
                  None => self.tcx.sess.span_bug(expr.span,
                                ~"Loop to unknown label")
              }
          }

          expr_assign(l, r) => {
            // see comment on lvalues in
            // propagate_through_lvalue_components()
            let succ = self.write_lvalue(l, succ, ACC_WRITE);
            let succ = self.propagate_through_lvalue_components(l, succ);
            self.propagate_through_expr(r, succ)
          }

          expr_swap(l, r) => {
            // see comment on lvalues in
            // propagate_through_lvalue_components()

            // I count swaps as `used` cause it might be something like:
            //    foo.bar <-> x
            // and I am too lazy to distinguish this case from
            //    y <-> x
            // (where both x, y are unused) just for a warning.
            let succ = self.write_lvalue(r, succ, ACC_WRITE|ACC_READ|ACC_USE);
            let succ = self.write_lvalue(l, succ, ACC_WRITE|ACC_READ|ACC_USE);
            let succ = self.propagate_through_lvalue_components(r, succ);
            self.propagate_through_lvalue_components(l, succ)
          }

          expr_assign_op(_, l, r) => {
            // see comment on lvalues in
            // propagate_through_lvalue_components()
            let succ = self.write_lvalue(l, succ, ACC_WRITE|ACC_READ);
            let succ = self.propagate_through_expr(r, succ);
            self.propagate_through_lvalue_components(l, succ)
          }

          // Uninteresting cases: just propagate in rev exec order

          expr_vstore(expr, _) => {
            self.propagate_through_expr(expr, succ)
          }

          expr_vec(exprs, _) => {
            self.propagate_through_exprs(exprs, succ)
          }

          expr_repeat(element, count, _) => {
            let succ = self.propagate_through_expr(count, succ);
            self.propagate_through_expr(element, succ)
          }

          expr_rec(fields, with_expr) => {
            let succ = self.propagate_through_opt_expr(with_expr, succ);
            do fields.foldr(succ) |field, succ| {
                self.propagate_through_expr(field.node.expr, succ)
            }
          }

          expr_struct(_, fields, with_expr) => {
            let succ = self.propagate_through_opt_expr(with_expr, succ);
            do fields.foldr(succ) |field, succ| {
                self.propagate_through_expr(field.node.expr, succ)
            }
          }

          expr_call(f, args, _) => {
            // calling a fn with bot return type means that the fn
            // will fail, and hence the successors can be ignored
            let t_ret = ty::ty_fn_ret(ty::expr_ty(self.tcx, f));
            let succ = if ty::type_is_bot(t_ret) {self.s.exit_ln}
                       else {succ};
            let succ = self.propagate_through_exprs(args, succ);
            self.propagate_through_expr(f, succ)
          }

          expr_tup(exprs) => {
            self.propagate_through_exprs(exprs, succ)
          }

          expr_binary(op, l, r) if ast_util::lazy_binop(op) => {
            let r_succ = self.propagate_through_expr(r, succ);

            let ln = self.live_node(expr.id, expr.span);
            self.init_from_succ(ln, succ);
            self.merge_from_succ(ln, r_succ, false);

            self.propagate_through_expr(l, ln)
          }

          expr_log(_, l, r) |
          expr_index(l, r) |
          expr_binary(_, l, r) => {
            self.propagate_through_exprs(~[l, r], succ)
          }

          expr_assert(e) |
          expr_addr_of(_, e) |
          expr_copy(e) |
          expr_unary_move(e) |
          expr_loop_body(e) |
          expr_do_body(e) |
          expr_cast(e, _) |
          expr_unary(_, e) |
          expr_paren(e) => {
            self.propagate_through_expr(e, succ)
          }

          expr_lit(*) => {
            succ
          }

          expr_block(blk) => {
            self.propagate_through_block(blk, succ)
          }

          expr_mac(*) => {
            self.tcx.sess.span_bug(expr.span, ~"unexpanded macro");
          }
        }
    }

    fn propagate_through_lvalue_components(expr: @expr,
                                           succ: LiveNode) -> LiveNode {
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
            expr_path(_) => succ,
            expr_field(e, _, _) => self.propagate_through_expr(e, succ),
            _ => self.propagate_through_expr(expr, succ)
        }
    }

    // see comment on propagate_through_lvalue()
    fn write_lvalue(expr: @expr,
                    succ: LiveNode,
                    acc: uint) -> LiveNode {
        match expr.node {
          expr_path(_) => self.access_path(expr, succ, acc),

          // We do not track other lvalues, so just propagate through
          // to their subcomponents.  Also, it may happen that
          // non-lvalues occur here, because those are detected in the
          // later pass borrowck.
          _ => succ
        }
    }

    fn access_path(expr: @expr, succ: LiveNode, acc: uint) -> LiveNode {
        let def = self.tcx.def_map.get(expr.id);
        match relevant_def(def) {
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

    fn propagate_through_loop(expr: @expr,
                              cond: Option<@expr>,
                              body: blk,
                              succ: LiveNode) -> LiveNode {

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
        debug!("propagate_through_loop: using id for loop body %d %s",
               expr.id, block_to_str(body, self.tcx.sess.intr()));

        let cond_ln = self.propagate_through_opt_expr(cond, ln);
        let body_ln = self.with_loop_nodes(expr.id, succ, ln, || {
            self.propagate_through_block(body, cond_ln)
        });

        // repeat until fixed point is reached:
        while self.merge_from_succ(ln, body_ln, first_merge) {
            first_merge = false;
            assert cond_ln == self.propagate_through_opt_expr(cond, ln);
            assert body_ln == self.with_loop_nodes(expr.id, succ, ln,
            || {
                self.propagate_through_block(body, cond_ln)
            });
        }

        cond_ln
    }

    fn with_loop_nodes<R>(loop_node_id: node_id,
                          break_ln: LiveNode,
                          cont_ln: LiveNode,
                          f: fn() -> R) -> R {
      debug!("with_loop_nodes: %d %u", loop_node_id, *break_ln);
        self.loop_scope.push(loop_node_id);
        self.break_ln.insert(loop_node_id, break_ln);
        self.cont_ln.insert(loop_node_id, cont_ln);
        let r = f();
        self.loop_scope.pop();
        move r
    }
}

// _______________________________________________________________________
// Checking for error conditions

fn check_local(local: @local, &&self: @Liveness, vt: vt<@Liveness>) {
    match local.node.init {
      Some(_) => {

        // Initializer:
        self.warn_about_unused_or_dead_vars_in_pat(local.node.pat);
        if !local.node.is_mutbl {
            self.check_for_reassignments_in_pat(local.node.pat);
        }
      }
      None => {

        // No initializer: the variable might be unused; if not, it
        // should not be live at this point.

        debug!("check_local() with no initializer");
        do self.pat_bindings(local.node.pat) |ln, var, sp| {
            if !self.warn_about_unused(sp, ln, var) {
                match self.live_on_exit(ln, var) {
                  None => { /* not live: good */ }
                  Some(lnk) => {
                    self.report_illegal_read(
                        local.span, lnk, var,
                        PossiblyUninitializedVariable);
                  }
                }
            }
        }
      }
    }

    visit::visit_local(local, self, vt);
}

fn check_arm(arm: arm, &&self: @Liveness, vt: vt<@Liveness>) {
    do self.arm_pats_bindings(arm.pats) |ln, var, sp| {
        self.warn_about_unused(sp, ln, var);
    }
    visit::visit_arm(arm, self, vt);
}

fn check_expr(expr: @expr, &&self: @Liveness, vt: vt<@Liveness>) {
    match expr.node {
      expr_path(_) => {
        for self.variable_from_def_map(expr.id, expr.span).each |var| {
            let ln = self.live_node(expr.id, expr.span);
            self.consider_last_use(expr, ln, *var);
        }

        visit::visit_expr(expr, self, vt);
      }

      expr_fn(*) | expr_fn_block(*) => {
        let caps = (*self.ir).captures(expr);
        for (*caps).each |cap| {
            let var = self.variable(cap.var_nid, expr.span);
            self.consider_last_use(expr, cap.ln, var);
            if cap.is_move {
                self.check_move_from_var(expr.span, cap.ln, var);
            }
        }

        visit::visit_expr(expr, self, vt);
      }

      expr_assign(l, r) => {
        self.check_lvalue(l, vt);
        vt.visit_expr(r, self, vt);

        visit::visit_expr(expr, self, vt);
      }

      expr_unary_move(r) => {
        self.check_move_from_expr(r, vt);

        visit::visit_expr(expr, self, vt);
      }

      expr_assign_op(_, l, _) => {
        self.check_lvalue(l, vt);

        visit::visit_expr(expr, self, vt);
      }

      expr_call(f, args, _) => {
        let targs = ty::ty_fn_args(ty::expr_ty(self.tcx, f));
        for vec::each2(args, targs) |arg_expr, arg_ty| {
            match ty::resolved_mode(self.tcx, arg_ty.mode) {
                by_val | by_copy | by_ref => {}
                by_move => {
                    if ty::expr_is_lval(self.tcx, self.ir.method_map,
                                        *arg_expr) {
                        // Probably a bad error message (what's an rvalue?)
                        // but I can't think of anything better
                        self.tcx.sess.span_err(arg_expr.span,
                          fmt!("Move mode argument must be an rvalue: try \
                          (move %s) instead", expr_to_str(*arg_expr,
                                                self.tcx.sess.intr())));
                    }
                }
            }
        }

        visit::visit_expr(expr, self, vt);
      }

      // no correctness conditions related to liveness
      expr_if(*) | expr_match(*) |
      expr_while(*) | expr_loop(*) |
      expr_index(*) | expr_field(*) | expr_vstore(*) |
      expr_vec(*) | expr_rec(*) | expr_tup(*) |
      expr_log(*) | expr_binary(*) |
      expr_assert(*) | expr_copy(*) |
      expr_loop_body(*) | expr_do_body(*) |
      expr_cast(*) | expr_unary(*) | expr_fail(*) |
      expr_ret(*) | expr_break(*) | expr_again(*) | expr_lit(_) |
      expr_block(*) | expr_swap(*) | expr_mac(*) | expr_addr_of(*) |
      expr_struct(*) | expr_repeat(*) | expr_paren(*) => {
        visit::visit_expr(expr, self, vt);
      }
    }
}

fn check_fn(_fk: visit::fn_kind, _decl: fn_decl,
            _body: blk, _sp: span, _id: node_id,
            &&_self: @Liveness, _v: vt<@Liveness>) {
    // do not check contents of nested fns
}

enum ReadKind {
    PossiblyUninitializedVariable,
    PossiblyUninitializedField,
    MovedVariable
}

impl @Liveness {
    fn check_ret(id: node_id, sp: span, _fk: visit::fn_kind,
                 entry_ln: LiveNode) {
        if self.live_on_entry(entry_ln, self.s.no_ret_var).is_some() {
            // if no_ret_var is live, then we fall off the end of the
            // function without any kind of return expression:

            let t_ret = ty::ty_fn_ret(ty::node_id_to_type(self.tcx, id));
            if ty::type_is_nil(t_ret) {
                // for nil return types, it is ok to not return a value expl.
            } else if ty::type_is_bot(t_ret) {
                // for bot return types, not ok.  Function should fail.
                self.tcx.sess.span_err(
                    sp, ~"some control paths may return");
            } else {
                self.tcx.sess.span_err(
                    sp, ~"not all control paths return a value");
            }
        }
    }

    /*
    Checks whether <var> is live on entry to any of the successors of <ln>.
    If it is, report an error.
    */
    fn check_move_from_var(span: span, ln: LiveNode, var: Variable) {
        debug!("check_move_from_var(%s, %s)",
               ln.to_str(), var.to_str());

        match self.live_on_exit(ln, var) {
          None => {}
          Some(lnk) => self.report_illegal_move(span, lnk, var)
        }
    }

    fn consider_last_use(expr: @expr, ln: LiveNode, var: Variable) {
        debug!("consider_last_use(expr.id=%?, ln=%s, var=%s)",
               expr.id, ln.to_str(), var.to_str());

        match self.live_on_exit(ln, var) {
          Some(_) => {}
          None => (*self.ir).add_last_use(expr.id, var)
       }
    }

    fn check_move_from_expr(expr: @expr, vt: vt<@Liveness>) {
        debug!("check_move_from_expr(node %d: %s)",
               expr.id, expr_to_str(expr, self.tcx.sess.intr()));

        if self.ir.method_map.contains_key(expr.id) {
            // actually an rvalue, since this calls a method
            return;
        }

        match expr.node {
          expr_path(_) => {
            match self.variable_from_path(expr) {
              Some(var) => {
                let ln = self.live_node(expr.id, expr.span);
                self.check_move_from_var(expr.span, ln, var);
              }
              None => {}
            }
          }

          expr_field(base, _, _) => {
            // Moving from x.y is allowed if x is never used later.
            // (Note that the borrowck guarantees that anything
            //  being moved from is uniquely tied to the stack frame)
            self.check_move_from_expr(base, vt);
          }

          expr_index(base, _) => {
            // Moving from x[y] is allowed if x is never used later.
            // (Note that the borrowck guarantees that anything
            //  being moved from is uniquely tied to the stack frame)
            self.check_move_from_expr(base, vt);
          }

          _ => {
            // For other kinds of lvalues, no checks are required,
            // and any embedded expressions are actually rvalues
          }
       }
    }

    fn check_lvalue(expr: @expr, vt: vt<@Liveness>) {
        match expr.node {
          expr_path(_) => {
            match self.tcx.def_map.get(expr.id) {
              def_local(nid, false) => {
                // Assignment to an immutable variable or argument:
                // only legal if there is no later assignment.
                let ln = self.live_node(expr.id, expr.span);
                let var = self.variable(nid, expr.span);
                self.check_for_reassignment(ln, var, expr.span);
                self.warn_about_dead_assign(expr.span, ln, var);
              }
              def => {
                match relevant_def(def) {
                  Some(nid) => {
                    let ln = self.live_node(expr.id, expr.span);
                    let var = self.variable(nid, expr.span);
                    self.warn_about_dead_assign(expr.span, ln, var);
                  }
                  None => {}
                }
              }
            }
          }

          _ => {
            // For other kinds of lvalues, no checks are required,
            // and any embedded expressions are actually rvalues
            visit::visit_expr(expr, self, vt);
          }
       }
    }

    fn check_for_reassignments_in_pat(pat: @pat) {
        do self.pat_bindings(pat) |ln, var, sp| {
            self.check_for_reassignment(ln, var, sp);
        }
    }

    fn check_for_reassignment(ln: LiveNode, var: Variable,
                              orig_span: span) {
        match self.assigned_on_exit(ln, var) {
          Some(ExprNode(span)) => {
            self.tcx.sess.span_err(
                span,
                ~"re-assignment of immutable variable");

            self.tcx.sess.span_note(
                orig_span,
                ~"prior assignment occurs here");
          }
          Some(lnk) => {
            self.tcx.sess.span_bug(
                orig_span,
                fmt!("illegal writer: %?", lnk));
          }
          None => {}
        }
    }

    fn report_illegal_move(move_span: span,
                           lnk: LiveNodeKind,
                           var: Variable) {

        // the only time that it is possible to have a moved variable
        // used by ExitNode would be arguments or fields in a ctor.
        // we give a slightly different error message in those cases.
        if lnk == ExitNode {
            let vk = self.ir.var_kinds[*var];
            match vk {
              Arg(_, name, _) => {
                self.tcx.sess.span_err(
                    move_span,
                    fmt!("illegal move from argument `%s`, which is not \
                          copy or move mode", self.tcx.sess.str_of(name)));
                return;
              }
              Self => {
                self.tcx.sess.span_err(
                    move_span,
                    ~"illegal move from self (cannot move out of a field of \
                       self)");
                return;
              }
              Local(*) | ImplicitRet => {
                self.tcx.sess.span_bug(
                    move_span,
                    fmt!("illegal reader (%?) for `%?`",
                         lnk, vk));
              }
            }
        }

        self.report_illegal_read(move_span, lnk, var, MovedVariable);
        self.tcx.sess.span_note(
            move_span, ~"move of variable occurred here");

    }

    fn report_illegal_read(chk_span: span,
                           lnk: LiveNodeKind,
                           var: Variable,
                           rk: ReadKind) {
        let msg = match rk {
          PossiblyUninitializedVariable => {
            ~"possibly uninitialized variable"
          }
          PossiblyUninitializedField => ~"possibly uninitialized field",
          MovedVariable => ~"moved variable"
        };
        let name = (*self.ir).variable_name(var);
        match lnk {
          FreeVarNode(span) => {
            self.tcx.sess.span_err(
                span,
                fmt!("capture of %s: `%s`", msg, name));
          }
          ExprNode(span) => {
            self.tcx.sess.span_err(
                span,
                fmt!("use of %s: `%s`", msg, name));
          }
          ExitNode |
          VarDefNode(_) => {
            self.tcx.sess.span_bug(
                chk_span,
                fmt!("illegal reader: %?", lnk));
          }
        }
    }

    fn should_warn(var: Variable) -> Option<~str> {
        let name = (*self.ir).variable_name(var);
        if name[0] == ('_' as u8) {None} else {Some(name)}
    }

    fn warn_about_unused_args(decl: fn_decl, entry_ln: LiveNode) {
        for decl.inputs.each |arg| {
            do pat_util::pat_bindings(self.tcx.def_map, arg.pat)
                    |_bm, p_id, sp, _n| {
                let var = self.variable(p_id, sp);
                self.warn_about_unused(sp, entry_ln, var);
            }
        }
    }

    fn warn_about_unused_or_dead_vars_in_pat(pat: @pat) {
        do self.pat_bindings(pat) |ln, var, sp| {
            if !self.warn_about_unused(sp, ln, var) {
                self.warn_about_dead_assign(sp, ln, var);
            }
        }
    }

    fn warn_about_unused(sp: span, ln: LiveNode, var: Variable) -> bool {
        if !self.used_on_entry(ln, var) {
            for self.should_warn(var).each |name| {

                // annoying: for parameters in funcs like `fn(x: int)
                // {ret}`, there is only one node, so asking about
                // assigned_on_exit() is not meaningful.
                let is_assigned = if ln == self.s.exit_ln {
                    false
                } else {
                    self.assigned_on_exit(ln, var).is_some()
                };

                if is_assigned {
                    // FIXME(#3266)--make liveness warnings lintable
                    self.tcx.sess.span_warn(
                        sp, fmt!("variable `%s` is assigned to, \
                                  but never used", *name));
                } else {
                    // FIXME(#3266)--make liveness warnings lintable
                    self.tcx.sess.span_warn(
                        sp, fmt!("unused variable: `%s`", *name));
                }
            }
            return true;
        }
        return false;
    }

    fn warn_about_dead_assign(sp: span, ln: LiveNode, var: Variable) {
        if self.live_on_exit(ln, var).is_none() {
            for self.should_warn(var).each |name| {
                // FIXME(#3266)--make liveness warnings lintable
                self.tcx.sess.span_warn(
                    sp,
                    fmt!("value assigned to `%s` is never read", *name));
            }
        }
    }
 }
