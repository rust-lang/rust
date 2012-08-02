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
 * # Extension to handle constructors
 *
 * Each field is assigned an index just as with local variables.  A use of
 * `self` is considered a use of all fields.  A use of `self.f` is just a use
 * of `f`.
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
 * - `reader`: the `live_node` ID of some node which will read the value
 *    that `V` holds on entry to `N`.  Formally: a node `M` such
 *    that there exists a path `P` from `N` to `M` where `P` does not
 *    write `V`.  If the `reader` is `invalid_node()`, then the current
 *    value will never be read (the variable is dead, essentially).
 *
 * - `writer`: the `live_node` ID of some node which will write the
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
 *
 * - `self_var`: a variable representing 'self'
 */

import dvec::{dvec, extensions};
import std::map::{hashmap, int_hash, str_hash, box_str_hash};
import syntax::{visit, ast_util};
import syntax::print::pprust::{expr_to_str};
import visit::vt;
import syntax::codemap::span;
import syntax::ast::*;
import driver::session::session;
import io::writer_util;
import capture::{cap_move, cap_drop, cap_copy, cap_ref};

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
type last_use_map = hashmap<node_id, @dvec<node_id>>;

enum variable = uint;
enum live_node = uint;

enum live_node_kind {
    lnk_freevar(span),
    lnk_expr(span),
    lnk_vdef(span),
    lnk_exit
}

fn check_crate(tcx: ty::ctxt,
               method_map: typeck::method_map,
               crate: @crate) -> last_use_map {
    let visitor = visit::mk_vt(@{
        visit_fn: visit_fn,
        visit_local: visit_local,
        visit_expr: visit_expr
        with *visit::default_visitor()
    });

    let last_use_map = int_hash();
    let initial_maps = @ir_maps(tcx, method_map,
                                last_use_map);
    visit::visit_crate(*crate, initial_maps, visitor);
    tcx.sess.abort_if_errors();
    return last_use_map;
}

impl of to_str::to_str for live_node {
    fn to_str() -> ~str { fmt!{"ln(%u)", *self} }
}

impl of to_str::to_str for variable {
    fn to_str() -> ~str { fmt!{"v(%u)", *self} }
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

impl methods for live_node {
    pure fn is_valid() -> bool { *self != uint::max_value }
}

fn invalid_node() -> live_node { live_node(uint::max_value) }

enum relevant_def { rdef_var(node_id), rdef_self }

type capture_info = {ln: live_node, is_move: bool, rv: relevant_def};

enum var_kind {
    vk_arg(node_id, ident, rmode),
    vk_local(node_id, ident),
    vk_field(ident),
    vk_self,
    vk_implicit_ret
}

fn relevant_def(def: def) -> option<relevant_def> {
    alt def {
      def_self(_) {some(rdef_self)}
      def_arg(nid, _) | def_local(nid, _) {some(rdef_var(nid))}
      _ {none}
    }
}

class ir_maps {
    let tcx: ty::ctxt;
    let method_map: typeck::method_map;
    let last_use_map: last_use_map;

    let mut num_live_nodes: uint;
    let mut num_vars: uint;
    let live_node_map: hashmap<node_id, live_node>;
    let variable_map: hashmap<node_id, variable>;
    let field_map: hashmap<ident, variable>;
    let capture_map: hashmap<node_id, @~[capture_info]>;
    let mut var_kinds: ~[var_kind];
    let mut lnks: ~[live_node_kind];

    new(tcx: ty::ctxt, method_map: typeck::method_map,
        last_use_map: last_use_map) {
        self.tcx = tcx;
        self.method_map = method_map;
        self.last_use_map = last_use_map;

        self.num_live_nodes = 0u;
        self.num_vars = 0u;
        self.live_node_map = int_hash();
        self.variable_map = int_hash();
        self.capture_map = int_hash();
        self.field_map = box_str_hash();
        self.var_kinds = ~[];
        self.lnks = ~[];
    }

    fn add_live_node(lnk: live_node_kind) -> live_node {
        let ln = live_node(self.num_live_nodes);
        vec::push(self.lnks, lnk);
        self.num_live_nodes += 1u;

        debug!{"%s is of kind %?", ln.to_str(), lnk};

        ln
    }

    fn add_live_node_for_node(node_id: node_id, lnk: live_node_kind) {
        let ln = self.add_live_node(lnk);
        self.live_node_map.insert(node_id, ln);

        debug!{"%s is node %d", ln.to_str(), node_id};
    }

    fn add_variable(vk: var_kind) -> variable {
        let v = variable(self.num_vars);
        vec::push(self.var_kinds, vk);
        self.num_vars += 1u;

        alt vk {
          vk_local(node_id, _) | vk_arg(node_id, _, _) {
            self.variable_map.insert(node_id, v);
          }
          vk_field(name) {
            self.field_map.insert(name, v);
          }
          vk_self | vk_implicit_ret {
          }
        }

        debug!{"%s is %?", v.to_str(), vk};

        v
    }

    fn variable(node_id: node_id, span: span) -> variable {
        alt self.variable_map.find(node_id) {
          some(var) {var}
          none {
            self.tcx.sess.span_bug(
                span, fmt!{"No variable registered for id %d", node_id});
          }
        }
    }

    fn variable_name(var: variable) -> ident {
        alt self.var_kinds[*var] {
          vk_local(_, name) | vk_arg(_, name, _) {name}
          vk_field(name) {@(~"self." + *name)}
          vk_self {@~"self"}
          vk_implicit_return {@~"<implicit-ret>"}
        }
    }

    fn set_captures(node_id: node_id, +cs: ~[capture_info]) {
        self.capture_map.insert(node_id, @cs);
    }

    fn captures(expr: @expr) -> @~[capture_info] {
        alt self.capture_map.find(expr.id) {
          some(caps) {caps}
          none {
            self.tcx.sess.span_bug(expr.span, ~"no registered caps");
          }
        }
    }

    fn lnk(ln: live_node) -> live_node_kind {
        self.lnks[*ln]
    }

    fn add_last_use(expr_id: node_id, var: variable) {
        let vk = self.var_kinds[*var];
        debug!{"Node %d is a last use of variable %?", expr_id, vk};
        alt vk {
          vk_arg(id, name, by_move) |
          vk_arg(id, name, by_copy) |
          vk_local(id, name) {
            let v = alt self.last_use_map.find(expr_id) {
              some(v) { v }
              none {
                let v = @dvec();
                self.last_use_map.insert(expr_id, v);
                v
              }
            };

            (*v).push(id);
          }
          vk_arg(_, _, by_ref) | vk_arg(_, _, by_mutbl_ref) |
          vk_arg(_, _, by_val) | vk_self | vk_field(_) | vk_implicit_ret {
            debug!{"--but it is not owned"};
          }
        }
    }
}

fn visit_fn(fk: visit::fn_kind, decl: fn_decl, body: blk,
            sp: span, id: node_id, &&self: @ir_maps, v: vt<@ir_maps>) {
    debug!{"visit_fn: id=%d", id};
    let _i = util::common::indenter();

    // swap in a new set of IR maps for this function body:
    let fn_maps = @ir_maps(self.tcx, self.method_map,
                           self.last_use_map);

    debug!{"creating fn_maps: %x", ptr::addr_of(*fn_maps) as uint};

    for decl.inputs.each |arg| {
        debug!{"adding argument %d", arg.id};
        let mode = ty::resolved_mode(self.tcx, arg.mode);
        (*fn_maps).add_variable(vk_arg(arg.id, arg.ident, mode));
    };

    // gather up the various local variables, significant expressions,
    // and so forth:
    visit::visit_fn(fk, decl, body, sp, id, fn_maps, v);

    alt fk {
      visit::fk_ctor(_, _, _, _, class_did) {
        add_class_fields(fn_maps, class_did);
      }
      _ {}
    }

    // Special nodes and variables:
    // - exit_ln represents the end of the fn, either by return or fail
    // - implicit_ret_var is a pseudo-variable that represents
    //   an implicit return
    let specials = {
        exit_ln: (*fn_maps).add_live_node(lnk_exit),
        fallthrough_ln: (*fn_maps).add_live_node(lnk_exit),
        no_ret_var: (*fn_maps).add_variable(vk_implicit_ret),
        self_var: (*fn_maps).add_variable(vk_self)
    };

    // compute liveness
    let lsets = @liveness(fn_maps, specials);
    let entry_ln = (*lsets).compute(decl, body);

    // check for various error conditions
    let check_vt = visit::mk_vt(@{
        visit_fn: check_fn,
        visit_local: check_local,
        visit_expr: check_expr
        with *visit::default_visitor()
    });
    check_vt.visit_block(body, lsets, check_vt);
    lsets.check_ret(id, sp, fk, entry_ln);
    lsets.check_fields(sp, entry_ln);
    lsets.warn_about_unused_args(sp, decl, entry_ln);
}

fn add_class_fields(self: @ir_maps, did: def_id) {
    for ty::lookup_class_fields(self.tcx, did).each |field_ty| {
        assert field_ty.id.crate == local_crate;
        let var = (*self).add_variable(vk_field(field_ty.ident));
        self.field_map.insert(field_ty.ident, var);
    }
}

fn visit_local(local: @local, &&self: @ir_maps, vt: vt<@ir_maps>) {
    let def_map = self.tcx.def_map;
    do pat_util::pat_bindings(def_map, local.node.pat) |p_id, sp, path| {
        debug!{"adding local variable %d", p_id};
        let name = ast_util::path_to_ident(path);
        (*self).add_live_node_for_node(p_id, lnk_vdef(sp));
        (*self).add_variable(vk_local(p_id, name));
    }
    visit::visit_local(local, self, vt);
}

fn visit_expr(expr: @expr, &&self: @ir_maps, vt: vt<@ir_maps>) {
    alt expr.node {
      // live nodes required for uses or definitions of variables:
      expr_path(_) {
        let def = self.tcx.def_map.get(expr.id);
        debug!{"expr %d: path that leads to %?", expr.id, def};
        if relevant_def(def).is_some() {
            (*self).add_live_node_for_node(expr.id, lnk_expr(expr.span));
        }
        visit::visit_expr(expr, self, vt);
      }
      expr_fn(_, _, _, cap_clause) |
      expr_fn_block(_, _, cap_clause) {
        // Make a live_node for each captured variable, with the span
        // being the location that the variable is used.  This results
        // in better error messages than just pointing at the closure
        // construction site.
        let proto = ty::ty_fn_proto(ty::expr_ty(self.tcx, expr));
        let cvs = capture::compute_capture_vars(self.tcx, expr.id,
                                                proto, cap_clause);
        let mut call_caps = ~[];
        for cvs.each |cv| {
            alt relevant_def(cv.def) {
              some(rv) {
                let cv_ln = (*self).add_live_node(lnk_freevar(cv.span));
                let is_move = alt cv.mode {
                  cap_move | cap_drop {true} // var must be dead afterwards
                  cap_copy | cap_ref {false} // var can still be used
                };
                vec::push(call_caps, {ln: cv_ln, is_move: is_move, rv: rv});
              }
              none {}
            }
        }
        (*self).set_captures(expr.id, call_caps);

        visit::visit_expr(expr, self, vt);
      }

      // live nodes required for interesting control flow:
      expr_if(*) | expr_alt(*) | expr_while(*) | expr_loop(*) {
        (*self).add_live_node_for_node(expr.id, lnk_expr(expr.span));
        visit::visit_expr(expr, self, vt);
      }
      expr_binary(op, _, _) if ast_util::lazy_binop(op) {
        (*self).add_live_node_for_node(expr.id, lnk_expr(expr.span));
        visit::visit_expr(expr, self, vt);
      }

      // otherwise, live nodes are not required:
      expr_index(*) | expr_field(*) | expr_vstore(*) |
      expr_vec(*) | expr_rec(*) | expr_call(*) | expr_tup(*) |
      expr_log(*) | expr_binary(*) |
      expr_assert(*) | expr_addr_of(*) | expr_copy(*) |
      expr_loop_body(*) | expr_do_body(*) | expr_cast(*) |
      expr_unary(*) | expr_fail(*) |
      expr_break | expr_again | expr_lit(_) | expr_ret(*) |
      expr_block(*) | expr_move(*) | expr_unary_move(*) | expr_assign(*) |
      expr_swap(*) | expr_assign_op(*) | expr_mac(*) | expr_struct(*) => {
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
    reader: live_node,
    writer: live_node,
    used: bool
};

fn invalid_users() -> users {
    {reader: invalid_node(), writer: invalid_node(), used: false}
}

type specials = {
    exit_ln: live_node,
    fallthrough_ln: live_node,
    no_ret_var: variable,
    self_var: variable
};

const ACC_READ: uint = 1u;
const ACC_WRITE: uint = 2u;
const ACC_USE: uint = 4u;

class liveness {
    let tcx: ty::ctxt;
    let ir: @ir_maps;
    let s: specials;
    let successors: ~[mut live_node];
    let users: ~[mut users];
    let mut break_ln: live_node;
    let mut cont_ln: live_node;

    new(ir: @ir_maps, specials: specials) {
        self.ir = ir;
        self.tcx = ir.tcx;
        self.s = specials;
        self.successors =
            vec::to_mut(
                vec::from_elem(self.ir.num_live_nodes,
                               invalid_node()));
        self.users =
            vec::to_mut(
                vec::from_elem(self.ir.num_live_nodes * self.ir.num_vars,
                               invalid_users()));
        self.break_ln = invalid_node();
        self.cont_ln = invalid_node();
    }

    // _______________________________________________________________________

    fn live_node(node_id: node_id, span: span) -> live_node {
        alt self.ir.live_node_map.find(node_id) {
          some(ln) {ln}
          none {
            // This must be a mismatch between the ir_map construction
            // above and the propagation code below; the two sets of
            // code have to agree about which AST nodes are worth
            // creating liveness nodes for.
            self.tcx.sess.span_bug(
                span, fmt!{"No live node registered for node %d",
                           node_id});
          }
        }
    }

    fn variable_from_rdef(rv: relevant_def, span: span) -> variable {
        alt rv {
          rdef_self {self.s.self_var}
          rdef_var(nid) {self.variable(nid, span)}
        }
    }

    fn variable_from_path(expr: @expr) -> option<variable> {
        alt expr.node {
          expr_path(_) {
            let def = self.tcx.def_map.get(expr.id);
            relevant_def(def).map(
                |rdef| self.variable_from_rdef(rdef, expr.span)
            )
          }
          _ {none}
        }
    }

    fn variable(node_id: node_id, span: span) -> variable {
        (*self.ir).variable(node_id, span)
    }

    fn variable_from_def_map(node_id: node_id,
                             span: span) -> option<variable> {
        alt self.tcx.def_map.find(node_id) {
          some(def) {
            relevant_def(def).map(
                |rdef| self.variable_from_rdef(rdef, span)
            )
          }
          none {
            self.tcx.sess.span_bug(
                span, ~"Not present in def map")
          }
        }
    }

    fn pat_bindings(pat: @pat, f: fn(live_node, variable, span)) {
        let def_map = self.tcx.def_map;
        do pat_util::pat_bindings(def_map, pat) |p_id, sp, _n| {
            let ln = self.live_node(p_id, sp);
            let var = self.variable(p_id, sp);
            f(ln, var, sp);
        }
    }

    fn idx(ln: live_node, var: variable) -> uint {
        *ln * self.ir.num_vars + *var
    }

    fn live_on_entry(ln: live_node, var: variable)
        -> option<live_node_kind> {

        assert ln.is_valid();
        let reader = self.users[self.idx(ln, var)].reader;
        if reader.is_valid() {some((*self.ir).lnk(reader))} else {none}
    }

    fn live_on_exit(ln: live_node, var: variable)
        -> option<live_node_kind> {

        self.live_on_entry(copy self.successors[*ln], var)
    }

    fn used_on_entry(ln: live_node, var: variable) -> bool {
        assert ln.is_valid();
        self.users[self.idx(ln, var)].used
    }

    fn assigned_on_entry(ln: live_node, var: variable)
        -> option<live_node_kind> {

        assert ln.is_valid();
        let writer = self.users[self.idx(ln, var)].writer;
        if writer.is_valid() {some((*self.ir).lnk(writer))} else {none}
    }

    fn assigned_on_exit(ln: live_node, var: variable)
        -> option<live_node_kind> {

        self.assigned_on_entry(copy self.successors[*ln], var)
    }

    fn indices(ln: live_node, op: fn(uint)) {
        let node_base_idx = self.idx(ln, variable(0u));
        for uint::range(0u, self.ir.num_vars) |var_idx| {
            op(node_base_idx + var_idx)
        }
    }

    fn indices2(ln: live_node, succ_ln: live_node,
                op: fn(uint, uint)) {
        let node_base_idx = self.idx(ln, variable(0u));
        let succ_base_idx = self.idx(succ_ln, variable(0u));
        for uint::range(0u, self.ir.num_vars) |var_idx| {
            op(node_base_idx + var_idx, succ_base_idx + var_idx);
        }
    }

    fn write_vars(wr: io::writer,
                  ln: live_node,
                  test: fn(uint) -> live_node) {
        let node_base_idx = self.idx(ln, variable(0u));
        for uint::range(0u, self.ir.num_vars) |var_idx| {
            let idx = node_base_idx + var_idx;
            if test(idx).is_valid() {
                wr.write_str(~" ");
                wr.write_str(variable(var_idx).to_str());
            }
        }
    }

    fn ln_str(ln: live_node) -> ~str {
        do io::with_str_writer |wr| {
            wr.write_str(~"[ln(");
            wr.write_uint(*ln);
            wr.write_str(~") of kind ");
            wr.write_str(fmt!{"%?", copy self.ir.lnks[*ln]});
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

    fn init_empty(ln: live_node, succ_ln: live_node) {
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

    fn init_from_succ(ln: live_node, succ_ln: live_node) {
        // more efficient version of init_empty() / merge_from_succ()
        self.successors[*ln] = succ_ln;
        self.indices2(ln, succ_ln, |idx, succ_idx| {
            self.users[idx] = self.users[succ_idx]
        });
        debug!{"init_from_succ(ln=%s, succ=%s)",
               self.ln_str(ln), self.ln_str(succ_ln)};
    }

    fn merge_from_succ(ln: live_node, succ_ln: live_node,
                       first_merge: bool) -> bool {
        if ln == succ_ln { return false; }

        let mut changed = false;
        do self.indices2(ln, succ_ln) |idx, succ_idx| {
            changed |= copy_if_invalid(copy self.users[succ_idx].reader,
                                       self.users[idx].reader);
            changed |= copy_if_invalid(copy self.users[succ_idx].writer,
                                       self.users[idx].writer);
            if self.users[succ_idx].used && !self.users[idx].used {
                self.users[idx].used = true;
                changed = true;
            }
        }

        debug!{"merge_from_succ(ln=%s, succ=%s, first_merge=%b, changed=%b)",
               ln.to_str(), self.ln_str(succ_ln), first_merge, changed};
        return changed;

        fn copy_if_invalid(src: live_node, &dst: live_node) -> bool {
            if src.is_valid() {
                if !dst.is_valid() {
                    dst = src;
                    return true;
                }
            }
            return false;
        }
    }

    // Indicates that a local variable was *defined*; we know that no
    // uses of the variable can precede the definition (resolve checks
    // this) so we just clear out all the data.
    fn define(writer: live_node, var: variable) {
        let idx = self.idx(writer, var);
        self.users[idx].reader = invalid_node();
        self.users[idx].writer = invalid_node();

        debug!{"%s defines %s (idx=%u): %s", writer.to_str(), var.to_str(),
               idx, self.ln_str(writer)};
    }

    // Either read, write, or both depending on the acc bitset
    fn acc(ln: live_node, var: variable, acc: uint) {
        let idx = self.idx(ln, var);
        let user = &mut self.users[idx];

        if (acc & ACC_WRITE) != 0u {
            user.reader = invalid_node();
            user.writer = ln;
        }

        // Important: if we both read/write, must do read second
        // or else the write will override.
        if (acc & ACC_READ) != 0u {
            user.reader = ln;
        }

        if (acc & ACC_USE) != 0u {
            self.users[idx].used = true;
        }

        debug!{"%s accesses[%x] %s: %s",
               ln.to_str(), acc, var.to_str(), self.ln_str(ln)};
    }

    // _______________________________________________________________________

    fn compute(decl: fn_decl, body: blk) -> live_node {
        // if there is a `break` or `again` at the top level, then it's
        // effectively a return---this only occurs in `for` loops,
        // where the body is really a closure.
        let entry_ln: live_node =
            self.with_loop_nodes(self.s.exit_ln, self.s.exit_ln, || {
                self.propagate_through_fn_block(decl, body)
            });

        // hack to skip the loop unless #debug is enabled:
        debug!{"^^ liveness computation results for body %d (entry=%s)",
               {
                   for uint::range(0u, self.ir.num_live_nodes) |ln_idx| {
                       #debug["%s", self.ln_str(live_node(ln_idx))];
                   }
                   body.node.id
               },
               entry_ln.to_str()};

        entry_ln
    }

    fn propagate_through_fn_block(decl: fn_decl, blk: blk) -> live_node {
        // inputs passed by & mode should be considered live on exit:
        for decl.inputs.each |arg| {
            alt ty::resolved_mode(self.tcx, arg.mode) {
              by_mutbl_ref | by_ref | by_val {
                // These are "non-owned" modes, so register a read at
                // the end.  This will prevent us from moving out of
                // such variables but also prevent us from registering
                // last uses and so forth.
                let var = self.variable(arg.id, blk.span);
                self.acc(self.s.exit_ln, var, ACC_READ);
              }
              by_move | by_copy {
                // These are owned modes.  If we don't use the
                // variable, nobody will.
              }
            }
        }

        // as above, the "self" variable is a non-owned variable
        self.acc(self.s.exit_ln, self.s.self_var, ACC_READ);

        // in a ctor, there is an implicit use of self.f for all fields f:
        for self.ir.field_map.each_value |var| {
            self.acc(self.s.exit_ln, var, ACC_READ|ACC_USE);
        }

        // the fallthrough exit is only for those cases where we do not
        // explicitly return:
        self.init_from_succ(self.s.fallthrough_ln, self.s.exit_ln);
        if blk.node.expr.is_none() {
            self.acc(self.s.fallthrough_ln, self.s.no_ret_var, ACC_READ)
        }

        self.propagate_through_block(blk, self.s.fallthrough_ln)
    }

    fn propagate_through_block(blk: blk, succ: live_node) -> live_node {
        let succ = self.propagate_through_opt_expr(blk.node.expr, succ);
        do blk.node.stmts.foldr(succ) |stmt, succ| {
            self.propagate_through_stmt(stmt, succ)
        }
    }

    fn propagate_through_stmt(stmt: @stmt, succ: live_node) -> live_node {
        alt stmt.node {
          stmt_decl(decl, _) {
            return self.propagate_through_decl(decl, succ);
          }

          stmt_expr(expr, _) | stmt_semi(expr, _) {
            return self.propagate_through_expr(expr, succ);
          }
        }
    }

    fn propagate_through_decl(decl: @decl, succ: live_node) -> live_node {
        alt decl.node {
          decl_local(locals) {
            do locals.foldr(succ) |local, succ| {
                self.propagate_through_local(local, succ)
            }
          }
          decl_item(_) {
            succ
          }
        }
    }

    fn propagate_through_local(local: @local, succ: live_node) -> live_node {
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

        let opt_init = local.node.init.map(|i| i.expr );
        let mut succ = self.propagate_through_opt_expr(opt_init, succ);
        do self.pat_bindings(local.node.pat) |ln, var, _sp| {
            self.init_from_succ(ln, succ);
            self.define(ln, var);
            succ = ln;
        }
        succ
    }

    fn propagate_through_exprs(exprs: ~[@expr],
                               succ: live_node) -> live_node {
        do exprs.foldr(succ) |expr, succ| {
            self.propagate_through_expr(expr, succ)
        }
    }

    fn propagate_through_opt_expr(opt_expr: option<@expr>,
                                  succ: live_node) -> live_node {
        do opt_expr.foldl(succ) |succ, expr| {
            self.propagate_through_expr(expr, succ)
        }
    }

    fn propagate_through_expr(expr: @expr, succ: live_node) -> live_node {
        alt expr.node {
          // Interesting cases with control flow or which gen/kill

          expr_path(_) {
            self.access_path(expr, succ, ACC_READ | ACC_USE)
          }

          expr_field(e, nm, _) {
            // If this is a reference to `self.f` inside of a ctor,
            // then we treat it as a read of that variable.
            // Otherwise, we ignore it and just propagate down to
            // process `e`.
            alt self.as_self_field(e, nm) {
              some((ln, var)) {
                self.init_from_succ(ln, succ);
                self.acc(ln, var, ACC_READ | ACC_USE);
                ln
              }
              none {
                self.propagate_through_expr(e, succ)
              }
            }
          }

          expr_fn(*) | expr_fn_block(*) {
            // the construction of a closure itself is not important,
            // but we have to consider the closed over variables.
            let caps = (*self.ir).captures(expr);
            do (*caps).foldr(succ) |cap, succ| {
                self.init_from_succ(cap.ln, succ);
                let var = self.variable_from_rdef(cap.rv, expr.span);
                self.acc(cap.ln, var, ACC_READ | ACC_USE);
                cap.ln
            }
          }

          expr_if(cond, then, els) {
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

          expr_while(cond, blk) {
            self.propagate_through_loop(expr, some(cond), blk, succ)
          }

          expr_loop(blk) {
            self.propagate_through_loop(expr, none, blk, succ)
          }

          expr_alt(e, arms, _) {
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
                let arm_succ =
                    self.propagate_through_opt_expr(
                        arm.guard,
                        self.propagate_through_block(arm.body, succ));
                self.merge_from_succ(ln, arm_succ, first_merge);
                first_merge = false;
            };
            self.propagate_through_expr(e, ln)
          }

          expr_ret(o_e) | expr_fail(o_e) { // ignore succ and subst exit_ln:
            self.propagate_through_opt_expr(o_e, self.s.exit_ln)
          }

          expr_break {
            if !self.break_ln.is_valid() {
                self.tcx.sess.span_bug(
                    expr.span, ~"break with invalid break_ln");
            }

            self.break_ln
          }

          expr_again {
            if !self.cont_ln.is_valid() {
                self.tcx.sess.span_bug(
                    expr.span, ~"cont with invalid cont_ln");
            }

            self.cont_ln
          }

          expr_move(l, r) | expr_assign(l, r) {
            // see comment on lvalues in
            // propagate_through_lvalue_components()
            let succ = self.write_lvalue(l, succ, ACC_WRITE);
            let succ = self.propagate_through_lvalue_components(l, succ);
            self.propagate_through_expr(r, succ)
          }

          expr_swap(l, r) {
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

          expr_assign_op(_, l, r) {
            // see comment on lvalues in
            // propagate_through_lvalue_components()
            let succ = self.write_lvalue(l, succ, ACC_WRITE|ACC_READ);
            let succ = self.propagate_through_expr(r, succ);
            self.propagate_through_lvalue_components(l, succ)
          }

          // Uninteresting cases: just propagate in rev exec order

          expr_vstore(expr, _) {
            self.propagate_through_expr(expr, succ)
          }

          expr_vec(exprs, _) {
            self.propagate_through_exprs(exprs, succ)
          }

          expr_rec(fields, with_expr) {
            let succ = self.propagate_through_opt_expr(with_expr, succ);
            do fields.foldr(succ) |field, succ| {
                self.propagate_through_expr(field.node.expr, succ)
            }
          }

          expr_struct(_, fields) {
            do fields.foldr(succ) |field, succ| {
                self.propagate_through_expr(field.node.expr, succ)
            }
          }

          expr_call(f, args, _) {
            // calling a fn with bot return type means that the fn
            // will fail, and hence the successors can be ignored
            let t_ret = ty::ty_fn_ret(ty::expr_ty(self.tcx, f));
            let succ = if ty::type_is_bot(t_ret) {self.s.exit_ln}
                       else {succ};
            let succ = self.propagate_through_exprs(args, succ);
            self.propagate_through_expr(f, succ)
          }

          expr_tup(exprs) {
            self.propagate_through_exprs(exprs, succ)
          }

          expr_binary(op, l, r) if ast_util::lazy_binop(op) {
            let r_succ = self.propagate_through_expr(r, succ);

            let ln = self.live_node(expr.id, expr.span);
            self.init_from_succ(ln, succ);
            self.merge_from_succ(ln, r_succ, false);

            self.propagate_through_expr(l, ln)
          }

          expr_log(_, l, r) |
          expr_index(l, r) |
          expr_binary(_, l, r) {
            self.propagate_through_exprs(~[l, r], succ)
          }

          expr_assert(e) |
          expr_addr_of(_, e) |
          expr_copy(e) |
          expr_unary_move(e) |
          expr_loop_body(e) |
          expr_do_body(e) |
          expr_cast(e, _) |
          expr_unary(_, e) {
            self.propagate_through_expr(e, succ)
          }

          expr_lit(*) {
            succ
          }

          expr_block(blk) {
            self.propagate_through_block(blk, succ)
          }

          expr_mac(*) {
            self.tcx.sess.span_bug(expr.span, ~"unexpanded macro");
          }
        }
    }

    fn propagate_through_lvalue_components(expr: @expr,
                                           succ: live_node) -> live_node {
        // # Lvalues
        //
        // In general, the full flow graph structure for an
        // assignment/move/etc can be handled in one of two ways,
        // depending on whether what is being assigned is a "tracked
        // value" or not. A tracked value is basically a local variable
        // or argument, or a self-field (`self.f`) in a ctor.
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
        // A tracked lvalue is either a local variable/argument `x` or
        // else it is a self-field `self.f` in a constructor.  In
        // these cases, the link_node where the write occurs is linked
        // to node id of `x` or `self`, respectively.  The
        // `write_lvalue()` routine generates the contents of this
        // node.  There are no subcomponents to consider.
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

        alt expr.node {
          expr_path(_) {
            succ
          }

          expr_field(e, nm, _) {
            alt self.as_self_field(e, nm) {
              some(_) {succ}
              none {self.propagate_through_expr(e, succ)}
            }
          }

          _ {
            self.propagate_through_expr(expr, succ)
          }
        }
    }

    // see comment on propagate_through_lvalue()
    fn write_lvalue(expr: @expr,
                    succ: live_node,
                    acc: uint) -> live_node {
        alt expr.node {
          expr_path(_) {
            self.access_path(expr, succ, acc)
          }

          expr_field(e, nm, _) {
            alt self.as_self_field(e, nm) {
              some((ln, var)) {
                self.init_from_succ(ln, succ);
                self.acc(ln, var, acc);
                ln
              }
              none {
                succ
              }
            }
          }

          // We do not track other lvalues, so just propagate through
          // to their subcomponents.  Also, it may happen that
          // non-lvalues occur here, because those are detected in the
          // later pass borrowck.
          _ {succ}
        }
    }

    fn access_path(expr: @expr, succ: live_node, acc: uint) -> live_node {
        let def = self.tcx.def_map.get(expr.id);
        alt relevant_def(def) {
          some(rdef_self) {
            // Accessing `self` is like accessing every field of
            // the current object. This allows something like
            // `self = ...;` (it will be considered a write to
            // every field, sensibly enough), though the borrowck
            // pass will reject it later on.
            //
            // Also, note that, within a ctor at least, an
            // expression like `self.f` is "shortcircuiting"
            // before it reaches this point by the code for
            // expr_field.
            let ln = self.live_node(expr.id, expr.span);
            if acc != 0u {
                self.init_from_succ(ln, succ);
                for self.ir.field_map.each_value |var| {
                    self.acc(ln, var, acc);
                }
            }
            ln
          }
          some(rdef_var(nid)) {
            let ln = self.live_node(expr.id, expr.span);
            if acc != 0u {
                self.init_from_succ(ln, succ);
                let var = self.variable(nid, expr.span);
                self.acc(ln, var, acc);
            }
            ln
          }
          none {
            succ
          }
        }
    }

    fn as_self_field(expr: @expr,
                     fld: ident) -> option<(live_node,variable)> {
        // If we checking a constructor, then we treat self.f as a
        // variable.  we use the live_node id that will be assigned to
        // the reference to self but the variable id for `f`.
        alt expr.node {
          expr_path(_) {
            let def = self.tcx.def_map.get(expr.id);
            alt def {
              def_self(_) {
                // Note: the field_map is empty unless we are in a ctor
                return self.ir.field_map.find(fld).map(|var| {
                    let ln = self.live_node(expr.id, expr.span);
                    (ln, var)
                });
              }
              _ { return none; }
            }
          }
          _ { return none; }
        }
    }

    fn propagate_through_loop(expr: @expr,
                              cond: option<@expr>,
                              body: blk,
                              succ: live_node) -> live_node {

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
        let cond_ln = self.propagate_through_opt_expr(cond, ln);
        let body_ln = self.with_loop_nodes(succ, ln, || {
            self.propagate_through_block(body, cond_ln)
        });

        // repeat until fixed point is reached:
        while self.merge_from_succ(ln, body_ln, first_merge) {
            first_merge = false;
            assert cond_ln == self.propagate_through_opt_expr(cond, ln);
            assert body_ln == self.with_loop_nodes(succ, ln, || {
                self.propagate_through_block(body, cond_ln)
            });
        }

        cond_ln
    }

    fn with_loop_nodes<R>(break_ln: live_node,
                          cont_ln: live_node,
                          f: fn() -> R) -> R {
        let bl = self.break_ln, cl = self.cont_ln;
        self.break_ln = break_ln;
        self.cont_ln = cont_ln;
        let r <- f();
        self.break_ln = bl;
        self.cont_ln = cl;
        return r;
    }
}

// _______________________________________________________________________
// Checking for error conditions

fn check_local(local: @local, &&self: @liveness, vt: vt<@liveness>) {
    alt local.node.init {
      some({op: op, expr: expr}) {

        // Initializer:

        alt op {
          init_move {self.check_move_from_expr(expr, vt)}
          init_assign {}
        }
        self.warn_about_unused_or_dead_vars_in_pat(local.node.pat);
        if !local.node.is_mutbl {
            self.check_for_reassignments_in_pat(local.node.pat);
        }
      }
      none {

        // No initializer: the variable might be unused; if not, it
        // should not be live at this point.

        debug!{"check_local() with no initializer"};
        do (*self).pat_bindings(local.node.pat) |ln, var, sp| {
            if !self.warn_about_unused(sp, ln, var) {
                alt (*self).live_on_exit(ln, var) {
                  none { /* not live: good */ }
                  some(lnk) {
                    self.report_illegal_read(
                        local.span, lnk, var,
                        possibly_uninitialized_variable);
                  }
                }
            }
        }
      }
    }

    visit::visit_local(local, self, vt);
}

fn check_expr(expr: @expr, &&self: @liveness, vt: vt<@liveness>) {
    alt expr.node {
      expr_path(_) {
        for (*self).variable_from_def_map(expr.id, expr.span).each |var| {
            let ln = (*self).live_node(expr.id, expr.span);
            self.consider_last_use(expr, ln, var);
        }

        visit::visit_expr(expr, self, vt);
      }

      expr_fn(_, _, _, cap_clause) | expr_fn_block(_, _, cap_clause) {
        let caps = (*self.ir).captures(expr);
        for (*caps).each |cap| {
            let var = (*self).variable_from_rdef(cap.rv, expr.span);
            self.consider_last_use(expr, cap.ln, var);
            if cap.is_move {
                self.check_move_from_var(expr.span, cap.ln, var);
            }
        }

        visit::visit_expr(expr, self, vt);
      }

      expr_assign(l, r) {
        self.check_lvalue(l, vt);
        vt.visit_expr(r, self, vt);

        visit::visit_expr(expr, self, vt);
      }

      expr_move(l, r) {
        self.check_lvalue(l, vt);
        self.check_move_from_expr(r, vt);

        visit::visit_expr(expr, self, vt);
      }

      expr_unary_move(r) {
        self.check_move_from_expr(r, vt);

        visit::visit_expr(expr, self, vt);
      }

      expr_assign_op(_, l, _) {
        self.check_lvalue(l, vt);

        visit::visit_expr(expr, self, vt);
      }

      expr_call(f, args, _) {
        let targs = ty::ty_fn_args(ty::expr_ty(self.tcx, f));
        vt.visit_expr(f, self, vt);
        do vec::iter2(args, targs) |arg_expr, arg_ty| {
            alt ty::resolved_mode(self.tcx, arg_ty.mode) {
              by_val | by_copy | by_ref | by_mutbl_ref{
                vt.visit_expr(arg_expr, self, vt);
              }
              by_move {
                self.check_move_from_expr(arg_expr, vt);
              }
            }
        }
      }

      // no correctness conditions related to liveness
      expr_if(*) | expr_alt(*) |
      expr_while(*) | expr_loop(*) |
      expr_index(*) | expr_field(*) | expr_vstore(*) |
      expr_vec(*) | expr_rec(*) | expr_tup(*) |
      expr_log(*) | expr_binary(*) |
      expr_assert(*) | expr_copy(*) |
      expr_loop_body(*) | expr_do_body(*) |
      expr_cast(*) | expr_unary(*) | expr_fail(*) |
      expr_ret(*) | expr_break | expr_again | expr_lit(_) |
      expr_block(*) | expr_swap(*) | expr_mac(*) | expr_addr_of(*) |
      expr_struct(*) {
        visit::visit_expr(expr, self, vt);
      }
    }
}

fn check_fn(_fk: visit::fn_kind, _decl: fn_decl,
            _body: blk, _sp: span, _id: node_id,
            &&_self: @liveness, _v: vt<@liveness>) {
    // do not check contents of nested fns
}

enum read_kind {
    possibly_uninitialized_variable,
    possibly_uninitialized_field,
    moved_variable
}

impl check_methods for @liveness {
    fn check_fields(sp: span, entry_ln: live_node) {
        for self.ir.field_map.each |nm, var| {
            alt (*self).live_on_entry(entry_ln, var) {
              none { /* ok */ }
              some(lnk_exit) {
                self.tcx.sess.span_err(
                    sp, fmt!{"field `self.%s` is never initialized", *nm});
              }
              some(lnk) {
                self.report_illegal_read(
                    sp, lnk, var, possibly_uninitialized_field);
              }
            }
        }
    }

    fn check_ret(id: node_id, sp: span, fk: visit::fn_kind,
                 entry_ln: live_node) {
        if (*self).live_on_entry(entry_ln, self.s.no_ret_var).is_some() {
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
                alt fk {
                  visit::fk_ctor(*) {
                    // ctors are written as though they are unit.
                  }
                  _ {
                    self.tcx.sess.span_err(
                        sp, ~"not all control paths return a value");
                  }
                }
            }
        }
    }

    fn check_move_from_var(span: span, ln: live_node, var: variable) {
        debug!{"check_move_from_var(%s, %s)",
               ln.to_str(), var.to_str()};

        alt (*self).live_on_exit(ln, var) {
          none { }
          some(lnk) {
            self.report_illegal_move(span, lnk, var);
          }
        }
    }

    fn consider_last_use(expr: @expr, ln: live_node, var: variable) {
        alt (*self).live_on_exit(ln, var) {
          some(_) {}
          none {
            (*self.ir).add_last_use(expr.id, var);
          }
       }
    }

    fn check_move_from_expr(expr: @expr, vt: vt<@liveness>) {
        debug!{"check_move_from_expr(node %d: %s)",
               expr.id, expr_to_str(expr)};

        if self.ir.method_map.contains_key(expr.id) {
            // actually an rvalue, since this calls a method
            return vt.visit_expr(expr, self, vt);
        }

        alt expr.node {
          expr_path(_) {
            alt (*self).variable_from_path(expr) {
              some(var) {
                let ln = (*self).live_node(expr.id, expr.span);
                self.check_move_from_var(expr.span, ln, var);
              }
              none {}
            }
          }

          expr_field(base, _, _) {
            // Moving from x.y is allowed if x is never used later.
            // (Note that the borrowck guarantees that anything
            //  being moved from is uniquely tied to the stack frame)
            self.check_move_from_expr(base, vt);
          }

          expr_index(base, idx) {
            // Moving from x[y] is allowed if x is never used later.
            // (Note that the borrowck guarantees that anything
            //  being moved from is uniquely tied to the stack frame)
            self.check_move_from_expr(base, vt);
            vt.visit_expr(idx, self, vt);
          }

          _ {
            // For other kinds of lvalues, no checks are required,
            // and any embedded expressions are actually rvalues
            vt.visit_expr(expr, self, vt);
          }
       }
    }

    fn check_lvalue(expr: @expr, vt: vt<@liveness>) {
        alt expr.node {
          expr_path(_) {
            alt self.tcx.def_map.get(expr.id) {
              def_local(nid, false) {
                // Assignment to an immutable variable or argument:
                // only legal if there is no later assignment.
                let ln = (*self).live_node(expr.id, expr.span);
                let var = (*self).variable(nid, expr.span);
                self.check_for_reassignment(ln, var, expr.span);
                self.warn_about_dead_assign(expr.span, ln, var);
              }
              def {
                alt relevant_def(def) {
                  some(rdef_var(nid)) {
                    let ln = (*self).live_node(expr.id, expr.span);
                    let var = (*self).variable(nid, expr.span);
                    self.warn_about_dead_assign(expr.span, ln, var);
                  }
                  some(rdef_self) {}
                  none {}
                }
              }
            }
          }

          _ {
            // For other kinds of lvalues, no checks are required,
            // and any embedded expressions are actually rvalues
            visit::visit_expr(expr, self, vt);
          }
       }
    }

    fn check_for_reassignments_in_pat(pat: @pat) {
        do (*self).pat_bindings(pat) |ln, var, sp| {
            self.check_for_reassignment(ln, var, sp);
        }
    }

    fn check_for_reassignment(ln: live_node, var: variable,
                              orig_span: span) {
        alt (*self).assigned_on_exit(ln, var) {
          some(lnk_expr(span)) {
            self.tcx.sess.span_err(
                span,
                ~"re-assignment of immutable variable");

            self.tcx.sess.span_note(
                orig_span,
                ~"prior assignment occurs here");
          }
          some(lnk) {
            self.tcx.sess.span_bug(
                orig_span,
                fmt!{"illegal writer: %?", lnk});
          }
          none {}
        }
    }

    fn report_illegal_move(move_span: span,
                           lnk: live_node_kind,
                           var: variable) {

        // the only time that it is possible to have a moved variable
        // used by lnk_exit would be arguments or fields in a ctor.
        // we give a slightly different error message in those cases.
        if lnk == lnk_exit {
            let vk = self.ir.var_kinds[*var];
            alt vk {
              vk_arg(_, name, _) {
                self.tcx.sess.span_err(
                    move_span,
                    fmt!{"illegal move from argument `%s`, which is not \
                          copy or move mode", *name});
                return;
              }
              vk_field(name) {
                self.tcx.sess.span_err(
                    move_span,
                    fmt!{"illegal move from field `%s`", *name});
                return;
              }
              vk_self {
                self.tcx.sess.span_err(
                    move_span,
                    ~"illegal move from self (cannot move out of a field of \
                       self)");
                return;
              }
              vk_local(*) | vk_implicit_ret {
                self.tcx.sess.span_bug(
                    move_span,
                    fmt!{"illegal reader (%?) for `%?`",
                         lnk, vk});
              }
            }
        }

        self.report_illegal_read(move_span, lnk, var, moved_variable);
        self.tcx.sess.span_note(
            move_span, ~"move of variable occurred here");

    }

    fn report_illegal_read(chk_span: span,
                           lnk: live_node_kind,
                           var: variable,
                           rk: read_kind) {
        let msg = alt rk {
          possibly_uninitialized_variable {~"possibly uninitialized variable"}
          possibly_uninitialized_field {~"possibly uninitialized field"}
          moved_variable {~"moved variable"}
        };
        let name = (*self.ir).variable_name(var);
        alt lnk {
          lnk_freevar(span) {
            self.tcx.sess.span_err(
                span,
                fmt!{"capture of %s: `%s`", msg, *name});
          }
          lnk_expr(span) {
            self.tcx.sess.span_err(
                span,
                fmt!{"use of %s: `%s`", msg, *name});
          }
          lnk_exit |
          lnk_vdef(_) {
            self.tcx.sess.span_bug(
                chk_span,
                fmt!{"illegal reader: %?", lnk});
          }
        }
    }

    fn should_warn(var: variable) -> option<ident> {
        let name = (*self.ir).variable_name(var);
        if (*name)[0] == ('_' as u8) {none} else {some(name)}
    }

    fn warn_about_unused_args(sp: span, decl: fn_decl, entry_ln: live_node) {
        for decl.inputs.each |arg| {
            let var = (*self).variable(arg.id, arg.ty.span);
            alt ty::resolved_mode(self.tcx, arg.mode) {
              by_mutbl_ref {
                // for mutable reference arguments, something like
                //    x = 1;
                // is not worth warning about, as it has visible
                // side effects outside the fn.
                alt (*self).assigned_on_entry(entry_ln, var) {
                  some(_) { /*ok*/ }
                  none {
                    // but if it is not written, it ought to be used
                    self.warn_about_unused(sp, entry_ln, var);
                  }
                }
              }
              by_val | by_ref | by_move | by_copy {
                self.warn_about_unused(sp, entry_ln, var);
              }
            }
        }
    }

    fn warn_about_unused_or_dead_vars_in_pat(pat: @pat) {
        do (*self).pat_bindings(pat) |ln, var, sp| {
            if !self.warn_about_unused(sp, ln, var) {
                self.warn_about_dead_assign(sp, ln, var);
            }
        }
    }

    fn warn_about_unused(sp: span, ln: live_node, var: variable) -> bool {
        if !(*self).used_on_entry(ln, var) {
            for self.should_warn(var).each |name| {

                // annoying: for parameters in funcs like `fn(x: int)
                // {ret}`, there is only one node, so asking about
                // assigned_on_exit() is not meaningful.
                let is_assigned = if ln == self.s.exit_ln {
                    false
                } else {
                    (*self).assigned_on_exit(ln, var).is_some()
                };

                if is_assigned {
                    self.tcx.sess.span_warn(
                        sp, fmt!{"variable `%s` is assigned to, \
                                  but never used", *name});
                } else {
                    self.tcx.sess.span_warn(
                        sp, fmt!{"unused variable: `%s`", *name});
                }
            }
            return true;
        }
        return false;
    }

    fn warn_about_dead_assign(sp: span, ln: live_node, var: variable) {
        if (*self).live_on_exit(ln, var).is_none() {
            for self.should_warn(var).each |name| {
                self.tcx.sess.span_warn(
                    sp,
                    fmt!{"value assigned to `%s` is never read", *name});
            }
        }
    }
 }
