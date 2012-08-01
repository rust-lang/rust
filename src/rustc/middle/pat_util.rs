import syntax::ast::*;
import syntax::ast_util;
import syntax::ast_util::{path_to_ident, respan, walk_pat};
import syntax::fold;
import syntax::fold::*;
import syntax::codemap::span;
import std::map::hashmap;

export pat_binding_ids, pat_bindings, pat_id_map;
export pat_is_variant;

type pat_id_map = std::map::hashmap<ident, node_id>;

// This is used because same-named variables in alternative patterns need to
// use the node_id of their namesake in the first pattern.
fn pat_id_map(dm: resolve3::DefMap, pat: @pat) -> pat_id_map {
    let map = std::map::box_str_hash();
    do pat_bindings(dm, pat) |p_id, _s, n| {
      map.insert(path_to_ident(n), p_id);
    };
    ret map;
}

fn pat_is_variant(dm: resolve3::DefMap, pat: @pat) -> bool {
    alt pat.node {
      pat_enum(_, _) { true }
      pat_ident(_, _, none) {
        alt dm.find(pat.id) {
          some(def_variant(_, _)) { true }
          _ { false }
        }
      }
      _ { false }
    }
}

fn pat_bindings(dm: resolve3::DefMap, pat: @pat,
                it: fn(node_id, span, @path)) {
    do walk_pat(pat) |p| {
        alt p.node {
          pat_ident(_, pth, _) if !pat_is_variant(dm, p) {
            it(p.id, p.span, pth);
          }
          _ {}
        }
    }
}

fn pat_binding_ids(dm: resolve3::DefMap, pat: @pat) -> ~[node_id] {
    let mut found = ~[];
    pat_bindings(dm, pat, |b_id, _sp, _pt| vec::push(found, b_id) );
    ret found;
}
