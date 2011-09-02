// A pass that annotates for each loops and functions with the free
// variables that they contain.

import std::map;
import std::map::*;
import std::option;
import std::int;
import std::str;
import std::option::*;
import syntax::ast;
import syntax::ast_util;
import syntax::visit;
import driver::session;
import middle::resolve;
import syntax::codemap::span;

export annotate_freevars;
export freevar_map;
export get_freevars;
export has_freevars;

// A vector of defs representing the free variables referred to in a function.
// (The def_upvar will already have been stripped).
type freevar_info = @[ast::def];
type freevar_map = hashmap<ast::node_id, freevar_info>;

// Searches through part of the AST for all references to locals or
// upvars in this frame and returns the list of definition IDs thus found.
// Since we want to be able to collect upvars in some arbitrary piece
// of the AST, we take a walker function that we invoke with a visitor
// in order to start the search.
fn collect_freevars(def_map: &resolve::def_map,
                    walker: &fn(&visit::vt<int>)) -> freevar_info {
    let seen = new_int_hash();
    let refs = @mutable [];

    fn ignore_item(_i: &@ast::item, _depth: &int, _v: &visit::vt<int>) {}

    let walk_expr = lambda(expr: &@ast::expr, depth: &int,
                           v: &visit::vt<int>) {
        alt expr.node {
          ast::expr_fn(f) {
            if f.proto == ast::proto_block ||
               f.proto == ast::proto_closure {
                visit::visit_expr(expr, depth + 1, v);
            }
          }
          ast::expr_for_each(dcl, x, b) {
            v.visit_local(dcl, depth, v);
            v.visit_expr(x, depth, v);
            v.visit_block(b, depth + 1, v);
          }
          ast::expr_path(path) {
            let def = def_map.get(expr.id), i = 0;
            while i < depth {
                alt {def} {
                  ast::def_upvar(_, inner, _) {
                    def = *inner;
                  }
                  _ { break; }
                }
                i += 1;
            }
            if i == depth { // Made it to end of loop
                let dnum = ast_util::def_id_of_def(def).node;
                if !seen.contains_key(dnum) {
                    *refs += [def];
                    seen.insert(dnum, ());
                }
            }
          }
          _ { visit::visit_expr(expr, depth, v); }
        }
    };

    walker(visit::mk_vt(@{visit_item: ignore_item,
                          visit_expr: walk_expr
                          with *visit::default_visitor()}));
    ret @*refs;
}

// Build a map from every function and for-each body to a set of the
// freevars contained in it. The implementation is not particularly
// efficient as it fully recomputes the free variables at every
// node of interest rather than building up the free variables in
// one pass. This could be improved upon if it turns out to matter.
fn annotate_freevars(def_map: &resolve::def_map,
                     crate: &@ast::crate) -> freevar_map {
    let freevars = new_int_hash();

    let walk_fn = lambda (f: &ast::_fn, tps: &[ast::ty_param], sp: &span,
                          i: &ast::fn_ident, nid: ast::node_id) {
        let start_walk = lambda (v: &visit::vt<int>) {
            v.visit_fn(f, tps, sp, i, nid, 1, v);
        };
        let vars = collect_freevars(def_map, start_walk);
        freevars.insert(nid, vars);
    };
    let walk_expr = lambda (expr: &@ast::expr) {
        alt expr.node {
          ast::expr_for_each(local, _, body) {
            let start_walk = lambda (v: &visit::vt<int>) {
                v.visit_block(body, 1, v);
            };
            let vars = collect_freevars(def_map, start_walk);
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

fn get_freevars(tcx: &ty::ctxt, fid: ast::node_id) -> freevar_info {
    alt tcx.freevars.find(fid) {
      none. {
        fail ~"get_freevars: " + int::str(fid)
            + ~" has no freevars";
      }
      some(d) { ret d; }
    }
}
fn has_freevars(tcx: &ty::ctxt, fid: ast::node_id) -> bool {
    ret std::vec::len(*get_freevars(tcx, fid)) != 0u;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
