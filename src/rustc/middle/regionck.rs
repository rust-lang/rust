/*
 * The region checking pass. Ensures that region-annotated pointers never
 * outlive their referents.
 */

import driver::session::session;
import middle::ty;
import std::map::hashmap;
import syntax::{ast, visit};

// An "extended region", which includes the ordinarily-unnamed reference-
// counted heap and exchange heap regions. This is used to detect borrowing.
enum region_ext {
    re_rc,
    re_exheap,
    re_region(ty::region)
}

type ctxt = {
    tcx: ty::ctxt,
    enclosing_block: option<ast::node_id>
};

fn check_expr(expr: @ast::expr, cx: ctxt, visitor: visit::vt<ctxt>) {
    ty::walk_ty(cx.tcx, ty::expr_ty(cx.tcx, expr)) { |t|
        alt ty::get(t).struct {
            ty::ty_rptr(region, _) {
                alt region {
                    ty::re_named(_) | ty::re_caller(_) { /* ok */ }
                    ty::re_block(rbi) {
                        let referent_block_id = rbi;
                        let enclosing_block_id = alt cx.enclosing_block {
                            none {
                                cx.tcx.sess.span_bug(expr.span, "block " +
                                                     "region type outside " +
                                                     "a block?!");
                            }
                            some(eb) { eb }
                        };

                        let parent_blocks = cx.tcx.region_map.parent_blocks;
                        while enclosing_block_id != referent_block_id {
                            if parent_blocks.contains_key(referent_block_id) {
                                referent_block_id =
                                    parent_blocks.get(referent_block_id);
                            } else {
                                cx.tcx.sess.span_err(expr.span,
                                                     "reference escapes " +
                                                     "its block");
                                break;
                            }
                        }
                    }
                }
            }
            _ { /* no-op */ }
        }
    }

    visit::visit_expr(expr, cx, visitor);
}

fn check_block(blk: ast::blk, cx: ctxt, visitor: visit::vt<ctxt>) {
    let new_cx: ctxt = { enclosing_block: some(blk.node.id) with cx };
    visit::visit_block(blk, new_cx, visitor);
}

fn check_crate(ty_cx: ty::ctxt, crate: @ast::crate) {
    let cx: ctxt = {tcx: ty_cx, enclosing_block: none};
    let visitor = visit::mk_vt(@{
        visit_expr: check_expr,
        visit_block: check_block
        with *visit::default_visitor()
    });
    visit::visit_crate(*crate, cx, visitor);
}

