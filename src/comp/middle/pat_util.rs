import syntax::ast::*;
import syntax::ast_util;
import syntax::ast_util::respan;
import syntax::fold;
import syntax::fold::*;

export normalize_arms;
export normalize_pat;
export normalize_pat_def_map;
export pat_binding_ids;
export pat_bindings;
export pat_id_map;
export path_to_ident;

fn normalize_pat_def_map(dm: resolve::def_map, p: @pat) -> @pat {
  // have to do it the hard way b/c ast fold doesn't pass around
  // node IDs. bother.
  alt p.node {
      pat_wild { p }
      pat_ident(_, none) { normalize_one(dm, p) }
      pat_ident(q, some(r)) {
        @{node: pat_ident(q, some(normalize_pat_def_map(dm, r)))
            with *p}
      }
      pat_tag(a_path, subs) {
        @{node: pat_tag(a_path,
           vec::map(subs, {|p| normalize_pat_def_map(dm, p)})) with *p}
      }
      pat_rec(field_pats, b) {
        @{node: pat_rec(vec::map(field_pats,
           {|fp| {pat: normalize_pat_def_map(dm, fp.pat) with fp}}), b)
            with *p}
      }
      pat_tup(subs) {
        @{node: pat_tup(vec::map(subs, {|p| normalize_pat_def_map(dm, p)}))
            with *p}
      }
      pat_box(q) {
        @{node: pat_box(normalize_pat_def_map(dm, q))
            with *p}
      }
      pat_uniq(q) {
        @{node: pat_uniq(normalize_pat_def_map(dm, q))
            with *p}
      }
      pat_lit(_) { p }
      pat_range(_,_) { p }
    }
}

fn normalize_one(dm: resolve::def_map, p: @pat) -> @pat {
    alt dm.find(p.id) {
        some(d) {
          alt p.node {
              pat_ident(tag_path, _) { @{id: p.id,
                    node: pat_tag(tag_path, []),
                    span: p.span} }
              _ { p }
          }
        }
        none { p }
    }
}

fn normalize_pat(tcx: ty::ctxt, p: @pat) -> @pat {
  normalize_pat_def_map(tcx.def_map, p)
}

fn normalize_arms(tcx: ty::ctxt, arms:[arm]) -> [arm] {
      vec::map(arms, {|a|
            {pats:
              vec::map(a.pats, {|p|
                    pat_util::normalize_pat(tcx, p)})
                with a}})
}

type pat_id_map = std::map::hashmap<str, node_id>;

// This is used because same-named variables in alternative patterns need to
// use the node_id of their namesake in the first pattern.
fn pat_id_map(tcx: ty::ctxt, pat: @pat) -> pat_id_map {
    let map = std::map::new_str_hash::<node_id>();
    pat_bindings(normalize_pat(tcx, pat)) {|bound|
        let name = path_to_ident(alt bound.node
           { pat_ident(n, _) { n } });
        map.insert(name, bound.id);
    };
    ret map;
}

// This does *not* normalize. The pattern should be already normalized
// if you want to get a normalized pattern out of it.
// Could return a constrained type in order to express that (future work)
fn pat_bindings(pat: @pat, it: fn(@pat)) {
  alt pat.node {
      pat_ident(_, option::none) { it(pat); }
      pat_ident(_, option::some(sub)) { it(pat); pat_bindings(sub, it); }
      pat_tag(_, sub) { for p in sub { pat_bindings(p, it); } }
      pat_rec(fields, _) { for f in fields { pat_bindings(f.pat, it); } }
      pat_tup(elts) { for elt in elts { pat_bindings(elt, it); } }
      pat_box(sub) { pat_bindings(sub, it); }
      pat_uniq(sub) { pat_bindings(sub, it); }
      pat_wild | pat_lit(_) | pat_range(_, _) { }
    }
}

fn pat_binding_ids(pat: @pat) -> [node_id] {
    let found = [];
    pat_bindings(pat) {|b| found += [b.id]; };
    ret found;
}

fn path_to_ident(p: @path) -> ident {
    alt vec::last(p.node.idents) {
        none { // sigh
          fail "Malformed path"; }
      some(i) { ret i; }
    }
}
