import syntax::ast::*;
import syntax::ast_util;
import syntax::ast_util::respan;
import syntax::fold;
import syntax::fold::*;
import syntax::codemap::span;
import std::map::hashmap;

export walk_pat;
export pat_binding_ids, pat_bindings, pat_id_map;
export pat_is_variant;
export path_to_ident;

type pat_id_map = std::map::hashmap<str, node_id>;

// This is used because same-named variables in alternative patterns need to
// use the node_id of their namesake in the first pattern.
fn pat_id_map(dm: resolve::def_map, pat: @pat) -> pat_id_map {
    let map = std::map::str_hash();
    pat_bindings(dm, pat) {|p_id, _s, n|
      map.insert(path_to_ident(n), p_id);
    };
    ret map;
}

fn pat_is_variant(dm: resolve::def_map, pat: @pat) -> bool {
    alt pat.node {
      pat_enum(_, _) { true }
      pat_ident(_, none) {
        alt dm.find(pat.id) {
          some(def_variant(_, _)) { true }
          _ { false }
        }
      }
      _ { false }
    }
}

// This does *not* normalize. The pattern should be already normalized
// if you want to get a normalized pattern out of it.
// Could return a constrained type in order to express that (future work)
fn pat_bindings(dm: resolve::def_map, pat: @pat,
                it: fn(node_id, span, @path)) {
    walk_pat(pat) {|p|
        alt p.node {
          pat_ident(pth, _) if !pat_is_variant(dm, p) {
            it(p.id, p.span, pth);
          }
          _ {}
        }
    }
}

fn walk_pat(pat: @pat, it: fn(@pat)) {
    it(pat);
    alt pat.node {
      pat_ident(pth, some(p)) { walk_pat(p, it); }
      pat_rec(fields, _) { for f in fields { walk_pat(f.pat, it); } }
      pat_enum(_, s) | pat_tup(s) { for p in s { walk_pat(p, it); } }
      pat_box(s) | pat_uniq(s) { walk_pat(s, it); }
      pat_wild | pat_lit(_) | pat_range(_, _) | pat_ident(_, none) {}
    }
}

fn pat_binding_ids(dm: resolve::def_map, pat: @pat) -> [node_id] {
    let found = [];
    pat_bindings(dm, pat) {|b_id, _sp, _pt| found += [b_id]; };
    ret found;
}

fn path_to_ident(p: @path) -> ident {
  assert (vec::is_not_empty(p.node.idents)); // should be a constraint on path
  vec::last(p.node.idents)
}
