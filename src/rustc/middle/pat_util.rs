use syntax::ast::*;
use syntax::ast_util;
use syntax::ast_util::{path_to_ident, respan, walk_pat};
use syntax::fold;
use syntax::fold::*;
use syntax::codemap::span;
use std::map::HashMap;

export pat_binding_ids, pat_bindings, pat_id_map, PatIdMap;
export pat_is_variant, pat_is_binding_or_wild;

type PatIdMap = std::map::HashMap<ident, node_id>;

// This is used because same-named variables in alternative patterns need to
// use the node_id of their namesake in the first pattern.
fn pat_id_map(dm: resolve::DefMap, pat: @pat) -> PatIdMap {
    let map = std::map::uint_hash();
    do pat_bindings(dm, pat) |_bm, p_id, _s, n| {
      map.insert(path_to_ident(n), p_id);
    };
    return map;
}

fn pat_is_variant(dm: resolve::DefMap, pat: @pat) -> bool {
    match pat.node {
      pat_enum(_, _) => true,
      pat_ident(_, _, None) => match dm.find(pat.id) {
        Some(def_variant(_, _)) => true,
        _ => false
      },
      _ => false
    }
}

fn pat_is_binding_or_wild(dm: resolve::DefMap, pat: @pat) -> bool {
    match pat.node {
        pat_ident(*) => !pat_is_variant(dm, pat),
        pat_wild => true,
        _ => false
    }
}

fn pat_bindings(dm: resolve::DefMap, pat: @pat,
                it: fn(binding_mode, node_id, span, @path)) {
    do walk_pat(pat) |p| {
        match p.node {
          pat_ident(binding_mode, pth, _) if !pat_is_variant(dm, p) => {
            it(binding_mode, p.id, p.span, pth);
          }
          _ => {}
        }
    }
}

fn pat_binding_ids(dm: resolve::DefMap, pat: @pat) -> ~[node_id] {
    let mut found = ~[];
    pat_bindings(dm, pat, |_bm, b_id, _sp, _pt| vec::push(found, b_id) );
    return found;
}
