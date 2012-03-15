/*
 * Region resolution. This pass runs before typechecking and resolves region
 * names to the appropriate block.
 */

import driver::session::session;
import middle::ty;
import syntax::{ast, visit};
import util::common::new_def_hash;

import std::list;
import std::list::list;
import std::map;
import std::map::hashmap;

/* Represents the type of the most immediate parent node. */
enum parent {
    pa_fn_item(ast::node_id),
    pa_block(ast::node_id),
    pa_nested_fn(ast::node_id),
    pa_item(ast::node_id),
    pa_crate
}

/* Records the binding site of a region name. */
type binding = {
    name: str,
    id: ast::def_id
};

type region_map = {
    /* Mapping from a block/function expression to its parent. */
    parents: hashmap<ast::node_id,ast::node_id>,
    /* Mapping from a region type in the AST to its resolved region. */
    ast_type_to_region: hashmap<ast::node_id,ty::region>,
    /* Mapping from a local variable to its containing block. */
    local_blocks: hashmap<ast::node_id,ast::node_id>,
    /* Mapping from a region name to its function. */
    region_name_to_fn: hashmap<ast::def_id,ast::node_id>,
    /* Mapping from an AST type node to the region that `&` resolves to. */
    ast_type_to_inferred_region: hashmap<ast::node_id,ty::region>,
    /* Mapping from a call site (or `bind` site) to its containing block. */
    call_site_to_block: hashmap<ast::node_id,ast::node_id>,
    /*
     * Mapping from an address-of operator or alt expression to its containing
     * block. This is used as the region if the operand is an rvalue.
     */
    rvalue_to_block: hashmap<ast::node_id,ast::node_id>
};

type ctxt = {
    sess: session,
    def_map: resolve::def_map,
    region_map: @region_map,
    mut bindings: @list<binding>,

    /*
     * A list of local IDs that will be parented to the next block we
     * traverse. This is used when resolving `alt` statements. Since we see
     * the pattern before the associated block, upon seeing a pattern we must
     * parent all the bindings in that pattern to the next block we see.
     */
    mut queued_locals: [ast::node_id],

    parent: parent,

    /* True if we're within the pattern part of an alt, false otherwise. */
    in_alt: bool,

    /*
     * Points to the site of the current typeclass implementation, or none if
     * we're outside one.
     */
    self_binding: option<ast::def_id>
};

fn region_to_scope(region_map: @region_map, region: ty::region)
        -> ast::node_id {
    ret alt region {
        ty::re_caller(def_id) | ty::re_self(def_id) { def_id.node }
        ty::re_named(def_id) { region_map.region_name_to_fn.get(def_id) }
        ty::re_block(node_id) { node_id }
        ty::re_param(_) { fail "unresolved region in region_to_scope" }
    };
}

// Returns true if `subscope` is equal to or is lexically nested inside
// `superscope` and false otherwise.
fn scope_contains(region_map: @region_map, superscope: ast::node_id,
                  subscope: ast::node_id) -> bool {
    let mut subscope = subscope;
    while superscope != subscope {
        alt region_map.parents.find(subscope) {
            none { ret false; }
            some(scope) { subscope = scope; }
        }
    }
    ret true;
}

fn get_inferred_region(cx: ctxt, sp: syntax::codemap::span) -> ty::region {
    // We infer to the caller region if we're at item scope
    // and to the block region if we're at block scope.
    //
    // TODO: What do we do if we're in an alt?

    ret alt cx.parent {
        pa_fn_item(item_id) | pa_nested_fn(item_id) {
            ty::re_caller({crate: ast::local_crate, node: item_id})
        }
        pa_block(block_id) { ty::re_block(block_id) }
        pa_item(_) { ty::re_param(0u) }
        pa_crate { cx.sess.span_bug(sp, "inferred region at crate level?!"); }
    }
}

fn resolve_ty(ty: @ast::ty, cx: ctxt, visitor: visit::vt<ctxt>) {
    let inferred_region = get_inferred_region(cx, ty.span);
    cx.region_map.ast_type_to_inferred_region.insert(ty.id, inferred_region);

    alt ty.node {
        ast::ty_rptr({id: region_id, node: node}, _) {
            alt node {
                ast::re_inferred { /* no-op */ }
                ast::re_self {
                    alt cx.self_binding {
                        some(def_id) {
                            let region = ty::re_self(def_id);
                            let rm = cx.region_map;
                            rm.ast_type_to_region.insert(region_id, region);
                        }
                        none {
                            cx.sess.span_err(ty.span,
                                             "the `self` region is not \
                                              allowed here");
                        }
                    }
                }
                ast::re_named(ident) {
                    // If at item scope, introduce or reuse a binding. If at
                    // block scope, require that the binding be introduced.
                    let bindings = cx.bindings;
                    let mut region;
                    alt list::find(*bindings, {|b| ident == b.name}) {
                        some(binding) { region = ty::re_named(binding.id); }
                        none {
                            let def_id = {crate: ast::local_crate,
                                          node: region_id};
                            let binding = {name: ident, id: def_id};
                            cx.bindings = @list::cons(binding, cx.bindings);
                            region = ty::re_named(def_id);

                            alt cx.parent {
                                pa_fn_item(fn_id) | pa_nested_fn(fn_id) {
                                    let rf = cx.region_map.region_name_to_fn;
                                    rf.insert(def_id, fn_id);
                                }
                                pa_item(_) {
                                    cx.sess.span_err(ty.span,
                                                     "named region not " +
                                                     "allowed in this " +
                                                     "context");
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
                        }
                    }

                    let ast_type_to_region = cx.region_map.ast_type_to_region;
                    ast_type_to_region.insert(region_id, region);
                }
            }
        }
        _ { /* nothing to do */ }
    }

    visit::visit_ty(ty, cx, visitor);
}

fn record_parent(cx: ctxt, child_id: ast::node_id) {
    alt cx.parent {
        pa_fn_item(parent_id) |
        pa_item(parent_id) |
        pa_block(parent_id) |
        pa_nested_fn(parent_id) {
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
        ast::expr_call(_, _, _) | ast::expr_bind(_, _) {
            // Record the block that this call appears in.
            alt cx.parent {
                pa_block(blk_id) {
                    cx.region_map.call_site_to_block.insert(expr.id, blk_id);
                }
                _ { cx.sess.span_bug(expr.span, "expr outside of block?!"); }
            }
            visit::visit_expr(expr, cx, visitor);
        }
        ast::expr_addr_of(_, subexpr) | ast::expr_alt(subexpr, _, _) {
            // Record the block that this expression appears in, in case the
            // operand is an rvalue.
            alt cx.parent {
                pa_block(blk_id) {
                    cx.region_map.rvalue_to_block.insert(subexpr.id, blk_id);
                }
                _ { cx.sess.span_bug(expr.span, "expr outside of block?!"); }
            }
            visit::visit_expr(expr, cx, visitor);
        }
        _ { visit::visit_expr(expr, cx, visitor); }
    }
}

fn resolve_local(local: @ast::local, cx: ctxt, visitor: visit::vt<ctxt>) {
    alt cx.parent {
        pa_block(blk_id) {
            cx.region_map.rvalue_to_block.insert(local.node.id, blk_id);
        }
        _ { cx.sess.span_bug(local.span, "local outside of block?!"); }
    }
    visit::visit_local(local, cx, visitor);
}

fn resolve_item(item: @ast::item, cx: ctxt, visitor: visit::vt<ctxt>) {
    // Items create a new outer block scope as far as we're concerned.
    let mut parent;
    let mut self_binding = cx.self_binding;
    alt item.node {
        ast::item_fn(_, _, _) | ast::item_enum(_, _) {
            parent = pa_fn_item(item.id);
        }
        ast::item_impl(_, _, _, _) {
            self_binding = some({crate: ast::local_crate, node: item.id});
            parent = pa_item(item.id);
        }
        _ { parent = pa_item(item.id); }
    };
    let new_cx: ctxt = {bindings: @list::nil,
                        parent: parent,
                        in_alt: false,
                        self_binding: self_binding
                        with cx};
    visit::visit_item(item, new_cx, visitor);
}

fn resolve_crate(sess: session, def_map: resolve::def_map, crate: @ast::crate)
        -> @region_map {
    let cx: ctxt = {sess: sess,
                    def_map: def_map,
                    region_map: @{parents: map::int_hash(),
                                  ast_type_to_region: map::int_hash(),
                                  local_blocks: map::int_hash(),
                                  region_name_to_fn: new_def_hash(),
                                  ast_type_to_inferred_region:
                                    map::int_hash(),
                                  call_site_to_block: map::int_hash(),
                                  rvalue_to_block: map::int_hash()},
                    mut bindings: @list::nil,
                    mut queued_locals: [],
                    parent: pa_crate,
                    in_alt: false,
                    self_binding: none};
    let visitor = visit::mk_vt(@{
        visit_block: resolve_block,
        visit_item: resolve_item,
        visit_ty: resolve_ty,
        visit_arm: resolve_arm,
        visit_pat: resolve_pat,
        visit_expr: resolve_expr,
        visit_local: resolve_local
        with *visit::default_visitor()
    });
    visit::visit_crate(*crate, cx, visitor);
    ret cx.region_map;
}

