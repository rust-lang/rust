import std::{smallintmap, option};
import syntax::ast::*;
import syntax::{visit, codemap};
import visit::vt;

tag ast_node {
    node_item(@item);
    node_obj_ctor(@item);
    node_native_item(@native_item);
    node_expr(@expr);
    node_arg(arg);
}

type map = std::map::hashmap<node_id, ast_node>;

fn map_crate(c: crate) -> map {
    // FIXME: This is using an adapter to convert the smallintmap
    // interface to the hashmap interface. It would be better to just
    // convert everything to use the smallintmap.
    let map = new_smallintmap_int_adapter::<ast_node>();

    let v_map = visit::mk_simple_visitor
        (@{visit_item: bind map_item(map, _),
           visit_native_item: bind map_native_item(map, _),
           visit_expr: bind map_expr(map, _),
           visit_fn: bind map_fn(map, _, _, _, _, _)
               with *visit::default_simple_visitor()});
    visit::visit_crate(c, (), v_map);
    ret map;
}

fn map_fn(map: map, f: _fn, _tp: [ty_param], _sp: codemap::span,
          _name: fn_ident, _id: node_id) {
    for a in f.decl.inputs { map.insert(a.id, node_arg(a)); }
}

fn map_item(map: map, i: @item) {
    map.insert(i.id, node_item(i));
    alt i.node {
      item_obj(_, _, ctor_id) { map.insert(ctor_id, node_obj_ctor(i)); }
      _ { }
    }
}

fn map_native_item(map: map, i: @native_item) {
    map.insert(i.id, node_native_item(i));
}

fn map_expr(map: map, ex: @expr) {
    map.insert(ex.id, node_expr(ex));
}

fn new_smallintmap_int_adapter<@V>() -> std::map::hashmap<int, V> {
    let key_idx = fn (key: int) -> uint { key as uint };
    let idx_key = fn (idx: uint) -> int { idx as int };
    ret new_smallintmap_adapter(key_idx, idx_key);
}

// This creates an object with the hashmap interface backed
// by the smallintmap type, because I don't want to go through
// the entire codebase adapting all the callsites to the different
// interface.
// FIXME: hashmap and smallintmap should support the same interface.
fn new_smallintmap_adapter<@K,
                           @V>(key_idx: fn(K) -> uint, idx_key: fn(uint) -> K)
   -> std::map::hashmap<K, V> {

    obj adapter<@K,
                @V>(map: smallintmap::smallintmap<V>,
                    key_idx: fn(K) -> uint,
                    idx_key: fn(uint) -> K) {

        fn size() -> uint { fail }

        fn insert(key: K, value: V) -> bool {
            let exists = smallintmap::contains_key(map, key_idx(key));
            smallintmap::insert(map, key_idx(key), value);
            ret !exists;
        }

        fn contains_key(key: K) -> bool {
            ret smallintmap::contains_key(map, key_idx(key));
        }

        fn get(key: K) -> V { ret smallintmap::get(map, key_idx(key)); }

        fn find(key: K) -> option::t<V> {
            ret smallintmap::find(map, key_idx(key));
        }

        fn remove(_key: K) -> option::t<V> { fail }

        fn rehash() { fail }

        iter items() -> @{key: K, val: V} {
            let idx = 0u;
            for item: option::t<V> in map.v {
                alt item {
                  option::some(elt) {
                    let value = elt;
                    let key = idx_key(idx);
                    put @{key: key, val: value};
                  }
                  option::none. { }
                }
                idx += 1u;
            }
        }
        iter keys() -> K {
            for each p: @{key: K, val: V} in self.items() { put p.key; }
        }
    }

    let map = smallintmap::mk::<V>();
    ret adapter(map, key_idx, idx_key);
}

fn node_span(node: ast_node) -> codemap::span {
    alt node {
      node_item(item) { item.span }
      node_obj_ctor(item) { item.span }
      node_native_item(nitem) { nitem.span }
      node_expr(expr) { expr.span }
    }
}

#[cfg(test)]
mod test {
    import syntax::ast_util;

    #[test]
    fn test_node_span_item() {
        let expected: codemap::span = ast_util::mk_sp(20u, 30u);
        let node =
            node_item(@{ident: "test",
                        attrs: [],
                        id: 0,
                        node: item_mod({view_items: [], items: []}),
                        span: expected});
        assert (node_span(node) == expected);
    }

    #[test]
    fn test_node_span_obj_ctor() {
        let expected: codemap::span = ast_util::mk_sp(20u, 30u);
        let node =
            node_obj_ctor(@{ident: "test",
                            attrs: [],
                            id: 0,
                            node: item_mod({view_items: [], items: []}),
                            span: expected});
        assert (node_span(node) == expected);
    }

    #[test]
    fn test_node_span_native_item() {
        let expected: codemap::span = ast_util::mk_sp(20u, 30u);
        let node =
            node_native_item(@{ident: "test",
                               attrs: [],
                               node: native_item_ty,
                               id: 0,
                               span: expected});
        assert (node_span(node) == expected);
    }

    #[test]
    fn test_node_span_expr() {
        let expected: codemap::span = ast_util::mk_sp(20u, 30u);
        let node = node_expr(@{id: 0, node: expr_break, span: expected});
        assert (node_span(node) == expected);
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
