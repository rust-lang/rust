// A pass that annotates for each loops and functions with the free
// variables that they contain.

import std::map;
import std::map::*;
import std::ivec;
import std::option;
import std::int;
import std::option::*;
import syntax::ast;
import syntax::visit;
import driver::session;
import middle::resolve;
import syntax::codemap::span;

export annotate_freevars;
export freevar_set;
export freevar_map;
export get_freevar_info;
export get_freevars;
export get_freevar_uses;
export has_freevars;
export is_freevar_of;
export def_lookup;

type freevar_set = hashset[ast::node_id];
type freevar_info = {defs: freevar_set, uses: @ast::node_id[]};
type freevar_map = hashmap[ast::node_id, freevar_info];

// Searches through part of the AST for all references to locals or
// upvars in this frame and returns the list of definition IDs thus found.
// Since we want to be able to collect upvars in some arbitrary piece
// of the AST, we take a walker function that we invoke with a visitor
// in order to start the search.
fn collect_freevars(def_map: &resolve::def_map, sess: &session::session,
                    walker: &fn(&visit::vt[()]) ,
                    initial_decls: ast::node_id[]) -> freevar_info {
    type env =
        @{mutable refs: ast::node_id[],
          decls: hashset[ast::node_id],
          def_map: resolve::def_map,
          sess: session::session};

    fn walk_fn(e: env, f: &ast::_fn, tps: &ast::ty_param[], sp: &span,
               i: &ast::fn_ident, nid: ast::node_id) {
        for a: ast::arg  in f.decl.inputs { e.decls.insert(a.id, ()); }
    }
    fn walk_expr(e: env, expr: &@ast::expr) {
        alt expr.node {
          ast::expr_path(path) {
            if !e.def_map.contains_key(expr.id) {
                e.sess.span_fatal(expr.span,
                                  "internal error in collect_freevars");
            }
            alt e.def_map.get(expr.id) {
              ast::def_arg(did) { e.refs += ~[expr.id]; }
              ast::def_local(did) { e.refs += ~[expr.id]; }
              ast::def_binding(did) { e.refs += ~[expr.id]; }
              _ {/* no-op */ }
            }
          }
          _ { }
        }
    }
    fn walk_local(e: env, local: &@ast::local) {
        set_add(e.decls, local.node.id);
    }
    fn walk_pat(e: env, p: &@ast::pat) {
        alt p.node { ast::pat_bind(_) { set_add(e.decls, p.id); } _ { } }
    }
    let decls: hashset[ast::node_id] = new_int_hash();
    for decl: ast::node_id  in initial_decls { set_add(decls, decl); }

    let e: env =
        @{mutable refs: ~[], decls: decls, def_map: def_map, sess: sess};
    walker(visit::mk_simple_visitor
           (@{visit_local: bind walk_local(e, _),
              visit_pat: bind walk_pat(e, _),
              visit_expr: bind walk_expr(e, _),
              visit_fn: bind walk_fn(e, _, _, _, _, _)
              with *visit::default_simple_visitor()}));

    // Calculate (refs - decls). This is the set of captured upvars.
    // We build a vec of the node ids of the uses and a set of the
    // node ids of the definitions.
    let uses = ~[];
    let defs = new_int_hash();
    for ref_id_: ast::node_id  in e.refs {
        let ref_id = ref_id_;
        let def_id = ast::def_id_of_def(def_map.get(ref_id)).node;
        if !decls.contains_key(def_id) {
            uses += ~[ref_id];
            set_add(defs, def_id);
        }
    }
    ret {defs: defs, uses: @uses};
}

// Build a map from every function and for-each body to a set of the
// freevars contained in it. The implementation is not particularly
// efficient as it fully recomputes the free variables at every
// node of interest rather than building up the free variables in
// one pass. This could be improved upon if it turns out to matter.
fn annotate_freevars(sess: &session::session, def_map: &resolve::def_map,
                     crate: &@ast::crate) -> freevar_map {
    type env =
        {freevars: freevar_map,
         def_map: resolve::def_map,
         sess: session::session};

    fn walk_fn(e: env, f: &ast::_fn, tps: &ast::ty_param[], sp: &span,
               i: &ast::fn_ident, nid: ast::node_id) {
        fn start_walk(f: &ast::_fn, tps: &ast::ty_param[], sp: &span,
                      i: &ast::fn_ident, nid: ast::node_id,
                      v: &visit::vt[()]) {
            v.visit_fn(f, tps, sp, i, nid, (), v);
        }
        let walker = bind start_walk(f, tps, sp, i, nid, _);
        let vars = collect_freevars(e.def_map, e.sess, walker, ~[]);
        e.freevars.insert(nid, vars);
    }
    fn walk_expr(e: env, expr: &@ast::expr) {
        alt expr.node {
          ast::expr_for_each(local, _, body) {
            fn start_walk(b: &ast::blk, v: &visit::vt[()]) {
                v.visit_block(b, (), v);
            }
            let vars =
                collect_freevars(e.def_map, e.sess, bind start_walk(body, _),
                                 ~[local.node.id]);
            e.freevars.insert(body.node.id, vars);
          }
          _ { }
        }
    }

    let e: env = {freevars: new_int_hash(), def_map: def_map, sess: sess};
    let visitor =
        visit::mk_simple_visitor(@{visit_fn: bind walk_fn(e, _, _, _, _, _),
                                   visit_expr: bind walk_expr(e, _)
                                      with *visit::default_simple_visitor()});
    visit::visit_crate(*crate, (), visitor);

    ret e.freevars;
}

fn get_freevar_info(tcx: &ty::ctxt, fid: ast::node_id) -> freevar_info {
    alt tcx.freevars.find(fid) {
      none. { fail "get_freevars: " + int::str(fid) + " has no freevars"; }
      some(d) { ret d; }
    }
}
fn get_freevars(tcx: &ty::ctxt, fid: ast::node_id) -> freevar_set {
    ret get_freevar_info(tcx, fid).defs;
}
fn get_freevar_uses(tcx: &ty::ctxt, fid: ast::node_id) -> @ast::node_id[] {
    ret get_freevar_info(tcx, fid).uses;
}
fn has_freevars(tcx: &ty::ctxt, fid: ast::node_id) -> bool {
    ret get_freevars(tcx, fid).size() != 0u;
}
fn is_freevar_of(tcx: &ty::ctxt, var: ast::node_id, f: ast::node_id) -> bool {
    ret get_freevars(tcx, f).contains_key(var);
}
fn def_lookup(tcx: &ty::ctxt, f: ast::node_id, id: ast::node_id) ->
   option::t[ast::def] {
    alt tcx.def_map.find(id) {
      none. { ret none; }
      some(d) {
        let did = ast::def_id_of_def(d);
        if is_freevar_of(tcx, did.node, f) {
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
