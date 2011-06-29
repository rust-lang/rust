import std::smallintmap;
import std::option;
import front::ast::*;
import visit::vt;

tag ast_node {
    node_item(@item);
    node_obj_ctor(@item);
    node_native_item(@native_item);
    node_expr(@expr);
}

type map = std::map::hashmap[node_id, ast_node];

fn map_crate(&crate c) -> map {
    // FIXME: This is using an adapter to convert the smallintmap
    // interface to the hashmap interface. It would be better to just
    // convert everything to use the smallintmap.
    auto map = new_smallintmap_int_adapter[ast_node]();

    auto v_map = @rec(visit_item=bind map_item(map, _, _, _),
                      visit_native_item=bind map_native_item(map, _, _, _),
                      visit_expr=bind map_expr(map, _, _, _)
                      with *visit::default_visitor[()]());
    visit::visit_crate(c, (), visit::vtor(v_map));
    ret map;
}

fn map_item(&map map, &@item i, &() e, &vt[()] v) {
    map.insert(i.id, node_item(i));
    alt (i.node) {
        case (item_obj(_, _, ?ctor_id)) {
            map.insert(ctor_id, node_obj_ctor(i));
        }
        case (_) {}
    }
    visit::visit_item(i, e, v);
}

fn map_native_item(&map map, &@native_item i, &() e, &vt[()] v) {
    map.insert(i.id, node_native_item(i));
    visit::visit_native_item(i, e, v);
}

fn map_expr(&map map, &@expr ex, &() e, &vt[()] v) {
    map.insert(ex.id, node_expr(ex));
    visit::visit_expr(ex, e, v);
}

fn new_smallintmap_int_adapter[V]() -> std::map::hashmap[int, V] {
    auto key_idx = fn(&int key) -> uint { key as uint };
    auto idx_key = fn(&uint idx) -> int { idx as int };
    ret new_smallintmap_adapter(key_idx, idx_key);
}

// This creates an object with the hashmap interface backed
// by the smallintmap type, because I don't want to go through
// the entire codebase adapting all the callsites to the different
// interface.
// FIXME: hashmap and smallintmap should support the same interface.
fn new_smallintmap_adapter[K, V](fn(&K) -> uint key_idx,
                                 fn(&uint) -> K idx_key)
    -> std::map::hashmap[K, V] {

    obj adapter[K, V](smallintmap::smallintmap[V] map,
                      fn(&K) -> uint key_idx,
                      fn(&uint) -> K idx_key) {

        fn size() -> uint { fail }

        fn insert(&K key, &V value) -> bool {
            auto exists = smallintmap::contains_key(map, key_idx(key));
            smallintmap::insert(map, key_idx(key), value);
            ret !exists;
        }

        fn contains_key(&K key) -> bool {
            ret smallintmap::contains_key(map, key_idx(key));
        }

        fn get(&K key) -> V {
            ret smallintmap::get(map, key_idx(key));
        }

        fn find(&K key) -> option::t[V] {
            ret smallintmap::find(map, key_idx(key));
        }

        fn remove(&K key) -> option::t[V] { fail }

        fn rehash() { fail }

        iter items() -> @tup(K, V) {
            auto idx = 0u;
            for (option::t[V] item in map.v) {
                alt (item) {
                    case (option::some(?elt)) {
                        auto value = elt;
                        auto key = idx_key(idx);
                        put @tup(key, value);
                    }
                    case (option::none) { }
                }
                idx += 1u;
            }
        }
    }

    auto map = smallintmap::mk[V]();
    ret adapter(map, key_idx, idx_key);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
