// A pass that annotates for each loops with the free variables that
// they contain.

import std::map;
import std::map::*;
import syntax::ast;
import syntax::walk;
import driver::session;
import middle::ty;
import syntax::codemap::span;



// Searches through part of the AST for all references to locals or
// upvars in this frame and returns the list of definition IDs thus found.
// Since we want to be able to collect upvars in some arbitrary piece
// of the AST, we take a walker function that we invoke with a visitor
// in order to start the search.
fn collect_upvars(&ty::ctxt tcx, &fn (&walk::ast_visitor) walker,
                  ast::node_id[] initial_decls) -> ast::node_id[] {
    type env =
        @rec(mutable ast::node_id[] refs,
             hashmap[ast::node_id, ()] decls,
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
                       "internal error in collect_upvars");
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
        e.decls.insert(local.node.id, ());
    }
    fn walk_pat(env e, &@ast::pat p) {
        alt (p.node) {
            case (ast::pat_bind(_)) {
                e.decls.insert(p.id, ());
            }
            case (_) {}
        }
    }
    let hashmap[ast::node_id, ()] decls = new_int_hash[()]();
    for (ast::node_id decl in initial_decls) { decls.insert(decl, ()); }

    let env e =
        @rec(mutable refs=~[],
             decls=decls,
             def_map=tcx.def_map,
             sess=tcx.sess);
    auto visitor =
        @rec(visit_fn_pre=bind walk_fn(e, _, _, _, _, _),
             visit_local_pre=bind walk_local(e, _),
             visit_expr_pre=bind walk_expr(e, _),
             visit_pat_pre=bind walk_pat(e, _)
             with walk::default_visitor());
    walker(*visitor);
    // Calculate (refs - decls). This is the set of captured upvars.

    let ast::node_id[] result = ~[];
    for (ast::node_id ref_id_ in e.refs) {
        auto ref_id = ref_id_;
        if (!decls.contains_key(ref_id)) { result += ~[ref_id]; }
    }
    ret result;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
