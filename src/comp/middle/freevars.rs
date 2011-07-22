// A pass that annotates for each loops and functions with the free
// variables that they contain.

import std::map;
import std::map::*;
import std::ivec;
import std::option;
import std::int;
import std::option::*;
import syntax::ast;
import syntax::walk;
import driver::session;
import middle::resolve;
import syntax::codemap::span;

export annotate_freevars;
export freevar_set;
export freevar_map;
export get_freevars;
export has_freevars;
export is_freevar_of;
export def_lookup;

type freevar_set = @ast::node_id[];
type freevar_map = hashmap[ast::node_id, freevar_set];

// Searches through part of the AST for all references to locals or
// upvars in this frame and returns the list of definition IDs thus found.
// Since we want to be able to collect upvars in some arbitrary piece
// of the AST, we take a walker function that we invoke with a visitor
// in order to start the search.
fn collect_freevars(&resolve::def_map def_map, &session::session sess,
                    &fn (&walk::ast_visitor) walker,
                    ast::node_id[] initial_decls) -> freevar_set {
    type env =
        @rec(mutable ast::node_id[] refs,
             hashset[ast::node_id] decls,
             resolve::def_map def_map,
             session::session sess);

    fn walk_fn(env e, &ast::_fn f, &ast::ty_param[] tps, &span sp,
               &ast::fn_ident i, ast::node_id nid) {
        for (ast::arg a in f.decl.inputs) { e.decls.insert(a.id, ()); }
    }
    fn walk_expr(env e, &@ast::expr expr) {
        alt (expr.node) {
            case (ast::expr_path(?path)) {
                if (! e.def_map.contains_key(expr.id)) {
                    e.sess.span_fatal(expr.span,
                       "internal error in collect_freevars");
                }
                alt (e.def_map.get(expr.id)) {
                    case (ast::def_arg(?did)) { e.refs += ~[did._1]; }
                    case (ast::def_local(?did)) { e.refs += ~[did._1]; }
                    case (ast::def_binding(?did)) { e.refs += ~[did._1]; }
                    case (_) { /* no-op */ }
                }
            }
            case (_) { }
        }
    }
    fn walk_local(env e, &@ast::local local) {
        set_add(e.decls, local.node.id);
    }
    fn walk_pat(env e, &@ast::pat p) {
        alt (p.node) {
            case (ast::pat_bind(_)) {
                set_add(e.decls, p.id);
            }
            case (_) {}
        }
    }
    let hashset[ast::node_id] decls = new_int_hash();
    for (ast::node_id decl in initial_decls) { set_add(decls, decl); }

    let env e =
        @rec(mutable refs=~[],
             decls=decls,
             def_map=def_map,
             sess=sess);
    auto visitor =
        @rec(visit_fn_pre=bind walk_fn(e, _, _, _, _, _),
             visit_local_pre=bind walk_local(e, _),
             visit_expr_pre=bind walk_expr(e, _),
             visit_pat_pre=bind walk_pat(e, _)
             with walk::default_visitor());
    walker(*visitor);

    // Calculate (refs - decls). This is the set of captured upvars.
    auto result = ~[];
    for (ast::node_id ref_id_ in e.refs) {
        auto ref_id = ref_id_;
        if (!decls.contains_key(ref_id)) { result += ~[ref_id]; }
    }
    ret @result;
}

// Build a map from every function and for-each body to a set of the
// freevars contained in it. The implementation is not particularly
// efficient as it fully recomputes the free variables at every
// node of interest rather than building up the free variables in
// one pass. This could be improved upon if it turns out to matter.
fn annotate_freevars(&session::session sess, &resolve::def_map def_map,
                     &@ast::crate crate) -> freevar_map {
    type env =
        rec(freevar_map freevars,
            resolve::def_map def_map,
            session::session sess);

    fn walk_fn(env e, &ast::_fn f, &ast::ty_param[] tps, &span sp,
               &ast::fn_ident i, ast::node_id nid) {
        auto walker = bind walk::walk_fn(_, f, tps, sp, i, nid);
        auto vars = collect_freevars(e.def_map, e.sess, walker, ~[]);
        e.freevars.insert(nid, vars);
    }
    fn walk_expr(env e, &@ast::expr expr) {
        alt (expr.node) {
            ast::expr_for_each(?local, _, ?body) {
                auto vars = collect_freevars(e.def_map, e.sess,
                                             bind walk::walk_block(_, body),
                                             ~[local.node.id]);
                e.freevars.insert(body.node.id, vars);
            }
            _ {}
        }
    }

    let env e =
        rec(freevars = new_int_hash(), def_map=def_map, sess=sess);
    auto visitor =
        rec(visit_fn_pre=bind walk_fn(e, _, _, _, _, _),
            visit_expr_pre=bind walk_expr(e, _)
            with walk::default_visitor());
    walk::walk_crate(visitor, *crate);

    ret e.freevars;
}

fn get_freevars(&ty::ctxt tcx, ast::node_id fid) -> freevar_set {
    alt (tcx.freevars.find(fid)) {
        none {
            fail "get_freevars: " + int::str(fid) + " has no freevars";
        }
        some(?d) { ret d; }
    }
}
fn has_freevars(&ty::ctxt tcx, ast::node_id fid) -> bool {
    ret ivec::len(*get_freevars(tcx, fid)) != 0u;
}
fn is_freevar_of(&ty::ctxt tcx, ast::node_id var, ast::node_id f) -> bool {
    ret ivec::member(var, *get_freevars(tcx, f));
}
fn def_lookup(&ty::ctxt tcx, ast::node_id f, ast::node_id id) ->
    option::t[ast::def] {
    alt (tcx.def_map.find(id)) {
      none { ret none; }
      some(?d) {
        auto did = ast::def_id_of_def(d);
        if is_freevar_of(tcx, did._1, f) {
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
