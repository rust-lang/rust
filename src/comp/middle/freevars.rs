// A pass that annotates for each loops and functions with the free
// variables that they contain.

import int;
import std::map::*;
import option::*;
import syntax::{ast, ast_util, visit};
import middle::resolve;
import syntax::codemap::span;

export annotate_freevars;
export freevar_map;
export freevar_info;
export get_freevars;
export has_freevars;

// A vector of defs representing the free variables referred to in a function.
// (The def_upvar will already have been stripped).
type freevar_entry = {
    def: ast::def, //< The variable being accessed free.
    span: span     //< First span where it is accessed (there can be multiple)
};
type freevar_info = @[@freevar_entry];
type freevar_map = hashmap<ast::node_id, freevar_info>;

// Searches through part of the AST for all references to locals or
// upvars in this frame and returns the list of definition IDs thus found.
// Since we want to be able to collect upvars in some arbitrary piece
// of the AST, we take a walker function that we invoke with a visitor
// in order to start the search.
fn collect_freevars(def_map: resolve::def_map, blk: ast::blk)
    -> freevar_info {
    let seen = new_int_hash();
    let refs = @mutable [];

    fn ignore_item(_i: @ast::item, &&_depth: int, _v: visit::vt<int>) { }

    let walk_expr =
        lambda (expr: @ast::expr, &&depth: int, v: visit::vt<int>) {
            alt expr.node {
              ast::expr_fn(decl, _, captures) {
                if decl.proto != ast::proto_bare {
                    visit::visit_expr(expr, depth + 1, v);
                }
              }
              ast::expr_fn_block(_, _) {
                visit::visit_expr(expr, depth + 1, v);
              }
              ast::expr_path(path) {
                let def = def_map.get(expr.id), i = 0;
                while i < depth {
                    alt copy def {
                      ast::def_upvar(_, inner, _) { def = *inner; }
                      _ { break; }
                    }
                    i += 1;
                }
                if i == depth { // Made it to end of loop
                    let dnum = ast_util::def_id_of_def(def).node;
                    if !seen.contains_key(dnum) {
                        *refs += [@{def:def, span:expr.span}];
                        seen.insert(dnum, ());
                    }
                }
              }
              _ { visit::visit_expr(expr, depth, v); }
            }
        };

    let v = visit::mk_vt(@{visit_item: ignore_item, visit_expr: walk_expr
                           with *visit::default_visitor()});
    v.visit_block(blk, 1, v);
    ret @*refs;
}

// Build a map from every function and for-each body to a set of the
// freevars contained in it. The implementation is not particularly
// efficient as it fully recomputes the free variables at every
// node of interest rather than building up the free variables in
// one pass. This could be improved upon if it turns out to matter.
fn annotate_freevars(def_map: resolve::def_map, crate: @ast::crate) ->
   freevar_map {
    let freevars = new_int_hash();

    let walk_fn = lambda (_decl: ast::fn_decl, _tps: [ast::ty_param],
                          blk: ast::blk, _sp: span, _nm: ast::fn_ident,
                          nid: ast::node_id) {
        let vars = collect_freevars(def_map, blk);
        freevars.insert(nid, vars);
    };

    let visitor =
        visit::mk_simple_visitor(@{visit_fn: walk_fn
                                   with *visit::default_simple_visitor()});
    visit::visit_crate(*crate, (), visitor);

    ret freevars;
}

fn get_freevars(tcx: ty::ctxt, fid: ast::node_id) -> freevar_info {
    alt tcx.freevars.find(fid) {
      none. { fail "get_freevars: " + int::str(fid) + " has no freevars"; }
      some(d) { ret d; }
    }
}
fn has_freevars(tcx: ty::ctxt, fid: ast::node_id) -> bool {
    ret vec::len(*get_freevars(tcx, fid)) != 0u;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
