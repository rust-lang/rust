/*
 * Region resolution. This pass runs before typechecking and resolves region
 * names to the appropriate block.
 */

import driver::session::session;
import middle::ty;
import syntax::{ast, visit};
import std::map;
import std::map::hashmap;

type region_map = {
    parent_blocks: hashmap<ast::node_id,ast::node_id>,
    ast_type_to_region: hashmap<ast::node_id,ty::region>
};

enum parent {
    pa_item(ast::node_id),
    pa_block(ast::node_id),
    pa_crate
}

type ctxt = {
    sess: session,
    region_map: @region_map,
    names_in_scope: hashmap<str,ast::def_id>,
    parent: parent
};

fn resolve_ty(ty: @ast::ty, cx: ctxt, visitor: visit::vt<ctxt>) {
    alt ty.node {
        ast::ty_rptr({id: region_id, node: node}, _) {
            let region;
            alt node {
                ast::re_inferred {
                    // We infer to the caller region if we're at item scope
                    // and to the block region if we're at block scope.
                    alt cx.parent {
                        pa_item(item_id) {
                            let def_id = {crate: ast::local_crate,
                                          node: item_id};
                            region = ty::re_caller(def_id);
                        }
                        pa_block(block_id) {
                            region = ty::re_block(block_id);
                        }
                        pa_crate {
                            cx.sess.span_bug(ty.span,
                                             "region type outside item");
                        }
                    }
                }

                ast::re_named(ident) {
                    // If at item scope, introduce or reuse a binding. If at
                    // block scope, require that the binding be introduced.
                    alt cx.names_in_scope.find(ident) {
                        some(def_id) { region = ty::re_named(def_id); }
                        none {
                            alt cx.parent {
                                pa_item(_) { /* ok; fall through */ }
                                pa_block(_) {
                                    cx.sess.span_err(ty.span,
                                                     "unknown region `" +
                                                     ident + "`");
                                }
                                pa_crate {
                                    cx.sess.span_bug(ty.span,
                                                     "named region at " +
                                                     "crate scope?!");
                                }
                            }

                            let def_id = {crate: ast::local_crate,
                                          node: region_id};
                            cx.names_in_scope.insert(ident, def_id);
                            region = ty::re_named(def_id);
                        }
                    }
                }

                ast::re_self {
                    // For blocks, "self" means "the current block".
                    alt cx.parent {
                        pa_item(_) {
                            cx.sess.span_unimpl(ty.span,
                                                "'self' region for items");
                        }
                        pa_block(block_id) {
                            region = ty::re_block(block_id);
                        }
                        pa_crate {
                            cx.sess.span_bug(ty.span,
                                             "region type outside item");
                        }
                    }
                }

            }

            cx.region_map.ast_type_to_region.insert(region_id, region);
        }
        _ { /* nothing to do */ }
    }

    visit::visit_ty(ty, cx, visitor);
}

fn resolve_block(blk: ast::blk, cx: ctxt, visitor: visit::vt<ctxt>) {
    alt cx.parent {
        pa_item(_) { /* no-op */ }
        pa_block(parent_block_id) {
            cx.region_map.parent_blocks.insert(blk.node.id, parent_block_id);
        }
        pa_crate { cx.sess.span_bug(blk.span, "block outside item?!"); }
    }

    let new_cx: ctxt = {parent: pa_block(blk.node.id) with cx};
    visit::visit_block(blk, new_cx, visitor);
}

fn resolve_item(item: @ast::item, cx: ctxt, visitor: visit::vt<ctxt>) {
    // Items create a new outer block scope as far as we're concerned.
    let new_cx: ctxt = {names_in_scope: map::new_str_hash(),
                        parent: pa_item(item.id)
                        with cx};
    visit::visit_item(item, new_cx, visitor);
}

fn resolve_crate(sess: session, crate: @ast::crate) -> @region_map {
    let cx: ctxt = {sess: sess,
                    region_map: @{parent_blocks: map::new_int_hash(),
                                  ast_type_to_region: map::new_int_hash()},
                    names_in_scope: map::new_str_hash(),
                    parent: pa_crate};
    let visitor = visit::mk_vt(@{
        visit_block: resolve_block,
        visit_item: resolve_item,
        visit_ty: resolve_ty
        with *visit::default_visitor()
    });
    visit::visit_crate(*crate, cx, visitor);
    ret cx.region_map;
}

