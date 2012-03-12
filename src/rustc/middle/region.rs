/*
 * Region resolution. This pass runs before typechecking and resolves region
 * names to the appropriate block.
 */

import driver::session::session;
import middle::ty;
import syntax::{ast, visit};
import std::map;
import std::map::hashmap;

/* Represents the type of the most immediate parent node. */
enum parent {
    pa_item(ast::node_id),
    pa_block(ast::node_id),
    pa_nested_fn(ast::node_id),
    pa_crate
}

type region_map = {
    /*
     * Mapping from blocks and function expression to their parent block or
     * function expression.
     */
    parents: hashmap<ast::node_id,ast::node_id>,
    /* Mapping from a region type in the AST to its resolved region. */
    ast_type_to_region: hashmap<ast::node_id,ty::region>,
    /* Mapping from a local variable to its containing block. */
    local_blocks: hashmap<ast::node_id,ast::node_id>,
};

type ctxt = {
    sess: session,
    def_map: resolve::def_map,
    region_map: @region_map,
    names_in_scope: hashmap<str,ast::def_id>,

    /*
     * A list of local IDs that will be parented to the next block we
     * traverse. This is used when resolving `alt` statements. Since we see
     * the pattern before the associated block, upon seeing a pattern we must
     * parent all the bindings in that pattern to the next block we see.
     */
    mut queued_locals: [ast::node_id],

    parent: parent,

    /* True if we're within the pattern part of an alt, false otherwise. */
    in_alt: bool
};

fn region_to_scope(_region_map: @region_map, region: ty::region)
        -> ast::node_id {
    ret alt region {
        ty::re_caller(def_id) { def_id.node }
        ty::re_named(_) { fail "TODO: named regions" }
        ty::re_block(node_id) { node_id }
    };
}

// Returns true if `subscope` is equal to or is lexically nested inside
// `superscope` and false otherwise.
fn scope_contains(region_map: @region_map, superscope: ast::node_id,
                  subscope: ast::node_id) -> bool {
    let subscope = subscope;
    while superscope != subscope {
        alt region_map.parents.find(subscope) {
            none { ret false; }
            some(scope) { subscope = scope; }
        }
    }
    ret true;
}

fn resolve_ty(ty: @ast::ty, cx: ctxt, visitor: visit::vt<ctxt>) {
    alt ty.node {
        ast::ty_rptr({id: region_id, node: node}, _) {
            let region;
            alt node {
                ast::re_inferred {
                    // We infer to the caller region if we're at item scope
                    // and to the block region if we're at block scope.
                    //
                    // TODO: What do we do if we're in an alt?

                    alt cx.parent {
                        pa_item(item_id) | pa_nested_fn(item_id) {
                            let def_id = {crate: ast::local_crate,
                                          node: item_id};
                            region = ty::re_caller(def_id);
                        }
                        pa_block(block_id) {
                            region = ty::re_block(block_id);
                        }
                        pa_crate {
                            cx.sess.span_bug(ty.span, "inferred region at " +
                                             "crate level?!");
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
                                pa_item(_) | pa_nested_fn(_) {
                                    /* ok; fall through */
                                }
                                pa_block(_) {
                                    cx.sess.span_err(ty.span,
                                                     "unknown region `" +
                                                     ident + "`");
                                }
                                pa_crate {
                                    cx.sess.span_bug(ty.span, "named " +
                                                     "region at crate " +
                                                     "level?!");
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
                    //
                    // TODO: What do we do in an alt?
                    //
                    // FIXME: Doesn't work in type items.

                    alt cx.parent {
                        pa_item(item_id) | pa_nested_fn(item_id) {
                            let def_id = {crate: ast::local_crate,
                                          node: item_id};
                            region = ty::re_caller(def_id);
                        }
                        pa_block(block_id) {
                            region = ty::re_block(block_id);
                        }
                        pa_crate {
                            cx.sess.span_bug(ty.span,
                                             "region type outside item?!");
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

fn record_parent(cx: ctxt, child_id: ast::node_id) {
    alt cx.parent {
        pa_item(parent_id) | pa_block(parent_id) | pa_nested_fn(parent_id) {
            cx.region_map.parents.insert(child_id, parent_id);
        }
        pa_crate { /* no-op */ }
    }
}

fn resolve_block(blk: ast::blk, cx: ctxt, visitor: visit::vt<ctxt>) {
    // Record the parent of this block.
    record_parent(cx, blk.node.id);

    // Resolve queued locals to this block.
    for local_id in cx.queued_locals {
        cx.region_map.local_blocks.insert(local_id, blk.node.id);
    }

    // Descend.
    let new_cx: ctxt = {parent: pa_block(blk.node.id),
                        mut queued_locals: [],
                        in_alt: false with cx};
    visit::visit_block(blk, new_cx, visitor);
}

fn resolve_arm(arm: ast::arm, cx: ctxt, visitor: visit::vt<ctxt>) {
    let new_cx: ctxt = {mut queued_locals: [], in_alt: true with cx};
    visit::visit_arm(arm, new_cx, visitor);
}

fn resolve_pat(pat: @ast::pat, cx: ctxt, visitor: visit::vt<ctxt>) {
    alt pat.node {
        ast::pat_ident(path, _) {
            let defn_opt = cx.def_map.find(pat.id);
            alt defn_opt {
                some(ast::def_variant(_,_)) {
                    /* Nothing to do; this names a variant. */
                }
                _ {
                    /*
                     * This names a local. Enqueue it or bind it to the
                     * containing block, depending on whether we're in an alt
                     * or not.
                     */
                    if cx.in_alt {
                        vec::push(cx.queued_locals, pat.id);
                    } else {
                        alt cx.parent {
                            pa_block(block_id) {
                                let local_blocks = cx.region_map.local_blocks;
                                local_blocks.insert(pat.id, block_id);
                            }
                            _ {
                                cx.sess.span_bug(pat.span,
                                                 "unexpected parent");
                            }
                        }
                    }
                }
            }
        }
        _ { /* no-op */ }
    }

    visit::visit_pat(pat, cx, visitor);
}

fn resolve_expr(expr: @ast::expr, cx: ctxt, visitor: visit::vt<ctxt>) {
    alt expr.node {
        ast::expr_fn(_, _, _, _) | ast::expr_fn_block(_, _) {
            record_parent(cx, expr.id);
            let new_cx = {parent: pa_nested_fn(expr.id),
                          in_alt: false with cx};
            visit::visit_expr(expr, new_cx, visitor);
        }
        _ { visit::visit_expr(expr, cx, visitor); }
    }
}

fn resolve_item(item: @ast::item, cx: ctxt, visitor: visit::vt<ctxt>) {
    // Items create a new outer block scope as far as we're concerned.
    let new_cx: ctxt = {names_in_scope: map::new_str_hash(),
                        parent: pa_item(item.id),
                        in_alt: false
                        with cx};
    visit::visit_item(item, new_cx, visitor);
}

fn resolve_crate(sess: session, def_map: resolve::def_map, crate: @ast::crate)
        -> @region_map {
    let cx: ctxt = {sess: sess,
                    def_map: def_map,
                    region_map: @{parents: map::new_int_hash(),
                                  ast_type_to_region: map::new_int_hash(),
                                  local_blocks: map::new_int_hash()},
                    names_in_scope: map::new_str_hash(),
                    mut queued_locals: [],
                    parent: pa_crate,
                    in_alt: false};
    let visitor = visit::mk_vt(@{
        visit_block: resolve_block,
        visit_item: resolve_item,
        visit_ty: resolve_ty,
        visit_arm: resolve_arm,
        visit_pat: resolve_pat,
        visit_expr: resolve_expr
        with *visit::default_visitor()
    });
    visit::visit_crate(*crate, cx, visitor);
    ret cx.region_map;
}

