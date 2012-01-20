import option;
import std::map;
import syntax::ast::*;
import syntax::ast_util;
import syntax::{visit, codemap};

enum ast_node {
    node_item(@item),
    node_native_item(@native_item),
    node_method(@method),
    node_expr(@expr),
    // Locals are numbered, because the alias analysis needs to know in which
    // order they are introduced.
    node_arg(arg, uint),
    node_local(uint),
    node_res_ctor(@item),
}

type map = std::map::map<node_id, ast_node>;
type ctx = @{map: map, mutable local_id: uint};

fn map_crate(c: crate) -> map {
    let cx = @{map: std::map::new_int_hash(),
               mutable local_id: 0u};

    let v_map = visit::mk_simple_visitor
        (@{visit_item: bind map_item(cx, _),
           visit_native_item: bind map_native_item(cx, _),
           visit_expr: bind map_expr(cx, _),
           visit_fn: bind map_fn(cx, _, _, _, _, _),
           visit_local: bind map_local(cx, _),
           visit_arm: bind map_arm(cx, _)
           with *visit::default_simple_visitor()});
    visit::visit_crate(c, (), v_map);
    ret cx.map;
}

fn map_fn(cx: ctx, _fk: visit::fn_kind, decl: fn_decl, _body: blk,
          _sp: codemap::span, _id: node_id) {
    for a in decl.inputs {
        cx.map.insert(a.id, node_arg(a, cx.local_id));
        cx.local_id += 1u;
    }
}

fn map_local(cx: ctx, loc: @local) {
    pat_util::pat_bindings(loc.node.pat) {|p|
        cx.map.insert(p.id, node_local(cx.local_id));
        cx.local_id += 1u;
    };
}

fn map_arm(cx: ctx, arm: arm) {
    pat_util::pat_bindings(arm.pats[0]) {|p|
        cx.map.insert(p.id, node_local(cx.local_id));
        cx.local_id += 1u;
    };
}

fn map_item(cx: ctx, i: @item) {
    cx.map.insert(i.id, node_item(i));
    alt i.node {
      item_impl(_, _, _, ms) {
        for m in ms { cx.map.insert(m.id, node_method(m)); }
      }
      item_res(_, _, _, dtor_id, ctor_id) {
        cx.map.insert(ctor_id, node_res_ctor(i));
        cx.map.insert(dtor_id, node_item(i));
      }
      _ { }
    }
}

fn map_native_item(cx: ctx, i: @native_item) {
    cx.map.insert(i.id, node_native_item(i));
}

fn map_expr(cx: ctx, ex: @expr) {
    cx.map.insert(ex.id, node_expr(ex));
}

fn node_span(node: ast_node) -> codemap::span {
    alt node {
      node_item(item) { item.span }
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
// End:
