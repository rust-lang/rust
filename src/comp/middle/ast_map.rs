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
    auto map = util::common::new_seq_int_hash[ast_node]();

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

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
