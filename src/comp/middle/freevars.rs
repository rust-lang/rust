// A pass that annotates for each loops and functions with the free
// variables that they contain.

import std::map;
import std::map::*;
import std::option;
import std::int;
import std::istr;
import std::option::*;
import syntax::ast;
import syntax::ast_util;
import syntax::visit;
import driver::session;
import middle::resolve;
import syntax::codemap::span;

export annotate_freevars;
export freevar_set;
export freevar_map;
export get_freevar_info;
export get_freevars;
export get_freevar_defs;
export has_freevars;
export is_freevar_of;
export def_lookup;

// Throughout the compiler, variables are generally dealt with using the
// node_ids of the reference sites and not the def_id of the definition
// site. Thus we store a set are the definitions along with a vec of one
// "canonical" referencing node_id per free variable. The set is useful for
// testing membership, the list of referencing sites is what you want for most
// other things.
type freevar_set = hashset<ast::node_id>;
type freevar_info = {defs: freevar_set, refs: @[ast::node_id]};
type freevar_map = hashmap<ast::node_id, freevar_info>;

// Searches through part of the AST for all references to locals or
// upvars in this frame and returns the list of definition IDs thus found.
// Since we want to be able to collect upvars in some arbitrary piece
// of the AST, we take a walker function that we invoke with a visitor
// in order to start the search.
fn collect_freevars(def_map: &resolve::def_map, sess: &session::session,
                    walker: &fn(&visit::vt<()>),
                    initial_decls: [ast::node_id]) -> freevar_info {
    let decls = new_int_hash();
    for decl: ast::node_id in initial_decls { set_add(decls, decl); }
    let refs = @mutable [];

    let walk_fn =
        lambda (f: &ast::_fn, _tps: &[ast::ty_param], _sp: &span,
                _i: &ast::fn_ident, _nid: ast::node_id) {
            for a: ast::arg in f.decl.inputs { set_add(decls, a.id); }
        };
    let walk_expr =
        lambda (expr: &@ast::expr) {
            alt expr.node {
              ast::expr_path(path) {
                if !def_map.contains_key(expr.id) {
                    sess.span_fatal(expr.span,
                                    ~"internal error in collect_freevars");
                }
                alt def_map.get(expr.id) {
                  ast::def_arg(did) { *refs += [expr.id]; }
                  ast::def_local(did) { *refs += [expr.id]; }
                  ast::def_binding(did) { *refs += [expr.id]; }
                  _ {/* no-op */ }
                }
              }
              _ { }
            }
        };
    let walk_local =
        lambda (local: &@ast::local) {
            for each b: @ast::pat in ast_util::pat_bindings(local.node.pat) {
                set_add(decls, b.id);
            }
        };
    let walk_pat =
        lambda (p: &@ast::pat) {
            alt p.node { ast::pat_bind(_) { set_add(decls, p.id); } _ { } }
        };

    walker(visit::mk_simple_visitor(@{visit_local: walk_local,
                                      visit_pat: walk_pat,
                                      visit_expr: walk_expr,
                                      visit_fn: walk_fn
                                         with
                                         *visit::default_simple_visitor()}));
    // Calculate (refs - decls). This is the set of captured upvars.
    // We build a vec of the node ids of the uses and a set of the
    // node ids of the definitions.
    let canonical_refs = [];
    let defs = new_int_hash();
    for ref_id_: ast::node_id in *refs {
        let ref_id = ref_id_;
        let def_id = ast_util::def_id_of_def(def_map.get(ref_id)).node;
        if !decls.contains_key(def_id) && !defs.contains_key(def_id) {
            canonical_refs += [ref_id];
            set_add(defs, def_id);
        }
    }
    ret {defs: defs, refs: @canonical_refs};
}

// Build a map from every function and for-each body to a set of the
// freevars contained in it. The implementation is not particularly
// efficient as it fully recomputes the free variables at every
// node of interest rather than building up the free variables in
// one pass. This could be improved upon if it turns out to matter.
fn annotate_freevars(sess: &session::session, def_map: &resolve::def_map,
                     crate: &@ast::crate) -> freevar_map {
    let freevars = new_int_hash();

    let walk_fn =
        lambda (f: &ast::_fn, tps: &[ast::ty_param], sp: &span,
                i: &ast::fn_ident, nid: ast::node_id) {
            let start_walk =
                lambda (v: &visit::vt<()>) {
                    v.visit_fn(f, tps, sp, i, nid, (), v);
                };
            let vars = collect_freevars(def_map, sess, start_walk, []);
            freevars.insert(nid, vars);
        };
    let walk_expr =
        lambda (expr: &@ast::expr) {
            alt expr.node {
              ast::expr_for_each(local, _, body) {
                let start_walk =
                    lambda (v: &visit::vt<()>) {
                        v.visit_block(body, (), v);
                    };
                let bound = ast_util::pat_binding_ids(local.node.pat);
                let vars = collect_freevars(def_map, sess, start_walk, bound);
                freevars.insert(body.node.id, vars);
              }
              _ { }
            }
        };

    let visitor =
        visit::mk_simple_visitor(@{visit_fn: walk_fn, visit_expr: walk_expr
                                      with *visit::default_simple_visitor()});
    visit::visit_crate(*crate, (), visitor);

    ret freevars;
}

fn get_freevar_info(tcx: &ty::ctxt, fid: ast::node_id) -> freevar_info {
    alt tcx.freevars.find(fid) {
      none. {
        fail "get_freevars: " + istr::to_estr(int::str(fid))
            + " has no freevars";
      }
      some(d) { ret d; }
    }
}
fn get_freevar_defs(tcx: &ty::ctxt, fid: ast::node_id) -> freevar_set {
    ret get_freevar_info(tcx, fid).defs;
}
fn get_freevars(tcx: &ty::ctxt, fid: ast::node_id) -> @[ast::node_id] {
    ret get_freevar_info(tcx, fid).refs;
}
fn has_freevars(tcx: &ty::ctxt, fid: ast::node_id) -> bool {
    ret get_freevar_defs(tcx, fid).size() != 0u;
}
fn is_freevar_of(tcx: &ty::ctxt, def: ast::node_id, f: ast::node_id) -> bool {
    ret get_freevar_defs(tcx, f).contains_key(def);
}
fn def_lookup(tcx: &ty::ctxt, f: ast::node_id, id: ast::node_id) ->
   option::t<ast::def> {
    alt tcx.def_map.find(id) {
      none. { ret none; }
      some(d) {
        let did = ast_util::def_id_of_def(d);
        if f != -1 && is_freevar_of(tcx, did.node, f) {
            ret some(ast::def_upvar(did, @d));
        } else { ret some(d); }
      }
    }
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
