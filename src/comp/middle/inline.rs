import std::map::hashmap;
import syntax::ast;
import syntax::ast_util;
import syntax::visit;
import middle::typeck::method_map;
import middle::trans::common::maps;
import metadata::csearch;

export inline_map;
export instantiate_inlines;

type inline_map = hashmap<ast::def_id, @ast::item>;

enum ctxt = {
    tcx: ty::ctxt,
    maps: maps,
    inline_map: inline_map,
    mutable to_process: [@ast::item]
};

fn instantiate_inlines(enabled: bool,
                       tcx: ty::ctxt,
                       maps: maps,
                       crate: @ast::crate) -> inline_map {
    let vt = visit::mk_vt(@{
        visit_expr: fn@(e: @ast::expr, cx: ctxt, vt: visit::vt<ctxt>) {
            visit::visit_expr(e, cx, vt);
            cx.visit_expr(e);
        }
        with *visit::default_visitor::<ctxt>()
    });
    let inline_map = ast_util::new_def_id_hash();
    let cx = ctxt({tcx: tcx, maps: maps,
                   inline_map: inline_map, mutable to_process: []});
    if enabled { visit::visit_crate(*crate, cx, vt); }
    while !vec::is_empty(cx.to_process) {
        let to_process = [];
        to_process <-> cx.to_process;
        #debug["Recursively looking at inlined items"];
        vec::iter(to_process, {|i| visit::visit_item(i, cx, vt)});
    }
    ret inline_map;
}

impl methods for ctxt {
    fn visit_expr(e: @ast::expr) {

        // Look for fn items or methods that are referenced which
        // ought to be inlined.

        alt e.node {
          ast::expr_path(_) {
            alt self.tcx.def_map.get(e.id) {
              ast::def_fn(did, _) {
                self.maybe_enqueue_fn(did);
              }
              _ { /* not a fn item, fallthrough */ }
            }
          }
          ast::expr_field(_, _, _) {
            alt self.maps.method_map.find(e.id) {
              some(origin) {
                self.maybe_enqueue_impl_method(origin);
              }
              _ { /* not an impl method, fallthrough */ }
            }
          }
          _ { /* fallthrough */ }
        }
    }

    fn maybe_enqueue_fn(did: ast::def_id) {
        if did.crate == ast::local_crate { ret; }
        if self.inline_map.contains_key(did) { ret; }
        alt csearch::maybe_get_item_ast(self.tcx, self.maps, did) {
          none {
            /* no AST attached, do not inline */
            #debug["No AST attached to def %s",
                   ty::item_path_str(self.tcx, did)];
          }
          some(item) { /* Found an AST, add to table: */
            #debug["Inlining def %s", ty::item_path_str(self.tcx, did)];
            self.to_process += [item];
            self.inline_map.insert(did, item);
          }
        }
    }

    fn maybe_enqueue_impl_method(_origin: typeck::method_origin) {
        // alt method_origin {
        //   method_static(did) { self.maybe_enqueue_fn(did); }
        //   method_param(_, _, _, _) | method_iface(_, _) {
        //     /* fallthrough */
        //   }
        // }
    }
}
