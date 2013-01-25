// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use middle::pat_util;
use middle::ty;
use middle::ty::{CopyValue, MoveValue, ReadValue, ValueMode, ctxt};
use middle::typeck::{method_map, method_map_entry};

use core::vec;
use std::map::HashMap;
use syntax::ast::{bind_infer, box, by_copy, by_move, by_ref, by_val, crate};
use syntax::ast::{deref, expr, expr_addr_of, expr_assign, expr_assign_op};
use syntax::ast::{expr_binary, expr_call, expr_copy, expr_field, expr_index};
use syntax::ast::{expr_match, expr_method_call, expr_paren, expr_path};
use syntax::ast::{expr_swap, expr_unary, neg, node_id, not, pat, pat_ident};
use syntax::ast::{expr_vstore, expr_vec, expr_rec, expr_tup, expr_lit};
use syntax::ast::{expr_cast, expr_if, expr_while, expr_loop, expr_fn};
use syntax::ast::{expr_fn_block, expr_loop_body, expr_do_body, expr_block};
use syntax::ast::{expr_unary_move, expr_fail, expr_break, expr_again};
use syntax::ast::{expr_ret, expr_log, expr_assert, expr_mac, expr_struct};
use syntax::ast::{expr_repeat};
use syntax::ast::{sty_uniq, sty_value, uniq};
use syntax::ast::{fn_decl, blk};
use syntax::visit;
use syntax::visit::{fn_kind, vt};
use syntax::print::pprust;
use syntax::codemap::span;

struct VisitContext {
    tcx: ctxt,
    method_map: HashMap<node_id,method_map_entry>,
    mode: ValueMode,
}

fn compute_modes_for_fn(fk: fn_kind,
                        decl: fn_decl,
                        body: blk,
                        sp: span,
                        id: node_id,
                        &&cx: VisitContext,
                        v: vt<VisitContext>) {
    let body_cx = VisitContext { mode: MoveValue, ..cx };
    visit::visit_fn(fk, decl, body, sp, id, body_cx, v);
}

fn compute_modes_for_fn_args(callee_id: node_id,
                             args: &[@expr],
                             last_arg_is_block: bool,
                             &&cx: VisitContext,
                             v: vt<VisitContext>) {
    let arg_tys = ty::ty_fn_args(ty::node_id_to_type(cx.tcx, callee_id));
    let mut i = 0;
    for vec::each2(args, arg_tys) |arg, arg_ty| {
        if last_arg_is_block && i == args.len() - 1 {
            let block_cx = VisitContext { mode: MoveValue, ..cx };
            compute_modes_for_expr(*arg, block_cx, v);
        } else {
            match ty::resolved_mode(cx.tcx, arg_ty.mode) {
                by_ref => {
                    let arg_cx = VisitContext { mode: ReadValue, ..cx };
                    compute_modes_for_expr(*arg, arg_cx, v);
                }
                by_val | by_move | by_copy => {
                    compute_modes_for_expr(*arg, cx, v);
                }
            }
        }
        i += 1;
    }
}

fn record_mode_for_expr(expr: @expr, &&cx: VisitContext) {
    match cx.mode {
        ReadValue | CopyValue => {
            cx.tcx.value_modes.insert(expr.id, cx.mode);
        }
        MoveValue => {
            // This is, contextually, a move, but if this expression
            // is implicitly copyable it's cheaper to copy.
            let e_ty = ty::expr_ty(cx.tcx, expr);
            if ty::type_implicitly_moves(cx.tcx, e_ty) {
                cx.tcx.value_modes.insert(expr.id, MoveValue);
            } else {
                cx.tcx.value_modes.insert(expr.id, CopyValue);
            }
        }
    }
}

fn compute_modes_for_expr(expr: @expr,
                          &&cx: VisitContext,
                          v: vt<VisitContext>) {
    debug!("compute_modes_for_expr(expr=%?/%s, mode=%?)",
           expr.id, pprust::expr_to_str(expr, cx.tcx.sess.intr()),
           cx.mode);

    // Adjust the mode if there was an implicit reference here.
    let cx = match cx.tcx.adjustments.find(expr.id) {
        None => cx,
        Some(adjustment) => {
            if adjustment.autoref.is_some() {
                VisitContext { mode: ReadValue, ..cx }
            } else {
                cx
            }
        }
    };

    match copy expr.node {
        expr_call(callee, args, is_block) => {
            let callee_cx = VisitContext { mode: ReadValue, ..cx };
            compute_modes_for_expr(callee, callee_cx, v);
            compute_modes_for_fn_args(callee.id, args, is_block, cx, v);
        }
        expr_path(*) => {
            record_mode_for_expr(expr, cx);
        }
        expr_copy(expr) => {
            let callee_cx = VisitContext { mode: CopyValue, ..cx };
            compute_modes_for_expr(expr, callee_cx, v);
        }
        expr_method_call(callee, _, _, args, is_block) => {
            // The LHS of the dot may or may not result in a move, depending
            // on the method map entry.
            let callee_mode;
            match cx.method_map.find(expr.id) {
                Some(ref method_map_entry) => {
                    match method_map_entry.explicit_self {
                        sty_uniq(_) | sty_value => callee_mode = MoveValue,
                        _ => callee_mode = ReadValue
                    }
                }
                None => {
                    cx.tcx.sess.span_bug(expr.span, ~"no method map entry");
                }
            }

            let callee_cx = VisitContext { mode: callee_mode, ..cx };
            compute_modes_for_expr(callee, callee_cx, v);

            compute_modes_for_fn_args(expr.callee_id, args, is_block, cx, v);
        }
        expr_binary(_, lhs, rhs) | expr_assign_op(_, lhs, rhs) => {
            // The signatures of these take their arguments by-ref, so they
            // don't copy or move.
            let arg_cx = VisitContext { mode: ReadValue, ..cx };
            compute_modes_for_expr(lhs, arg_cx, v);
            compute_modes_for_expr(rhs, arg_cx, v);
        }
        expr_addr_of(_, arg) => {
            // Takes its argument by-ref, so it doesn't copy or move.
            let arg_cx = VisitContext { mode: ReadValue, ..cx };
            compute_modes_for_expr(arg, arg_cx, v);
        }
        expr_unary(unop, arg) => {
            match unop {
                deref => {
                    // Derefs function as reads.
                    let arg_cx = VisitContext { mode: ReadValue, ..cx };
                    compute_modes_for_expr(arg, arg_cx, v);

                    // This is an lvalue, so it needs a value mode recorded
                    // for it.
                    record_mode_for_expr(expr, cx);
                }
                box(_) | uniq(_) => {
                    let arg_cx = VisitContext { mode: MoveValue, ..cx };
                    compute_modes_for_expr(arg, arg_cx, v);
                }
                not | neg => {
                    // Takes its argument by ref.
                    let arg_cx = VisitContext { mode: ReadValue, ..cx };
                    compute_modes_for_expr(arg, arg_cx, v);
                }
            }
        }
        expr_field(arg, _, _) => {
            let arg_cx = VisitContext { mode: ReadValue, ..cx };
            compute_modes_for_expr(arg, arg_cx, v);

            record_mode_for_expr(expr, cx);
        }
        expr_assign(lhs, rhs) => {
            // The signatures of these take their arguments by-ref, so they
            // don't copy or move.
            let arg_cx = VisitContext { mode: ReadValue, ..cx };
            compute_modes_for_expr(lhs, arg_cx, v);
            compute_modes_for_expr(rhs, cx, v);
        }
        expr_swap(lhs, rhs) => {
            let arg_cx = VisitContext { mode: ReadValue, ..cx };
            compute_modes_for_expr(lhs, arg_cx, v);
            compute_modes_for_expr(rhs, arg_cx, v);
        }
        expr_index(lhs, rhs) => {
            let lhs_cx = VisitContext { mode: ReadValue, ..cx };
            compute_modes_for_expr(lhs, lhs_cx, v);
            let rhs_cx = VisitContext { mode: MoveValue, ..cx };
            compute_modes_for_expr(rhs, rhs_cx, v);

            record_mode_for_expr(expr, cx);
        }
        expr_paren(arg) => {
            compute_modes_for_expr(arg, cx, v);
            record_mode_for_expr(expr, cx);
        }
        expr_match(head, ref arms) => {
            // We must do this first so that `arms_have_by_move_bindings`
            // below knows which bindings are moves.
            for arms.each |arm| {
                (v.visit_arm)(*arm, cx, v);
            }

            let by_move_bindings_present =
                pat_util::arms_have_by_move_bindings(cx.tcx, *arms);
            if by_move_bindings_present {
                // Propagate the current mode flag downward.
                visit::visit_expr(expr, cx, v);
            } else {
                // We aren't moving into any pattern, so this is just a read.
                let head_cx = VisitContext { mode: ReadValue, ..cx };
                compute_modes_for_expr(head, head_cx, v);
            }
        }
        // Spell out every remaining expression so we don't forget to
        // update this code if we add a new variant.
        // (Maybe a macro to do this would be nice...)
        expr_vstore(*) | expr_vec(*) | expr_rec(*) | expr_tup(*) |
            expr_lit(*) | expr_cast(*) | expr_if(*) | expr_while(*) |
            expr_loop(*) | expr_fn(*) | expr_fn_block(*) |
            expr_loop_body(*) | expr_do_body(*) | expr_block(*) |
            expr_unary_move(*) | expr_fail(*) | expr_break(*) |
            expr_again(*) | expr_ret(*) | expr_log(*) | expr_assert(*) |
            expr_mac(*) | expr_struct(*) | expr_repeat(*) => {
            visit::visit_expr(expr, cx, v)
        }
    }
}

fn compute_modes_for_pat(pat: @pat,
                         &&cx: VisitContext,
                         v: vt<VisitContext>) {
    match pat.node {
        pat_ident(bind_infer, _, _)
                if pat_util::pat_is_binding(cx.tcx.def_map, pat) => {
            if ty::type_implicitly_moves(cx.tcx, ty::pat_ty(cx.tcx, pat)) {
                cx.tcx.value_modes.insert(pat.id, MoveValue);
            } else {
                cx.tcx.value_modes.insert(pat.id, CopyValue);
            }
        }
        _ => {}
    }

    visit::visit_pat(pat, cx, v);
}

pub fn compute_modes(tcx: ctxt, method_map: method_map, crate: @crate) {
    let visitor = visit::mk_vt(@visit::Visitor {
        visit_fn: compute_modes_for_fn,
        visit_expr: compute_modes_for_expr,
        visit_pat: compute_modes_for_pat,
        .. *visit::default_visitor()
    });
    let callee_cx = VisitContext {
        tcx: tcx,
        method_map: method_map,
        mode: MoveValue
    };
    visit::visit_crate(*crate, callee_cx, visitor);
}

