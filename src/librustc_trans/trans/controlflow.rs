// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::ValueRef;
use middle::def;
use middle::lang_items::{PanicFnLangItem, PanicBoundsCheckFnLangItem};
use trans::base::*;
use trans::basic_block::BasicBlock;
use trans::build::*;
use trans::callee;
use trans::cleanup::CleanupMethods;
use trans::cleanup;
use trans::common::*;
use trans::consts;
use trans::debuginfo;
use trans::debuginfo::{DebugLoc, ToDebugLoc};
use trans::expr;
use trans::machine;
use trans;
use middle::ty;

use rustc_front::hir;
use rustc_front::util as ast_util;

use syntax::ast;
use syntax::parse::token::InternedString;
use syntax::parse::token;

pub fn trans_stmt<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                              s: &hir::Stmt)
                              -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_stmt");
    let fcx = cx.fcx;
    debug!("trans_stmt({:?})", s);

    if cx.unreachable.get() {
        return cx;
    }

    if cx.sess().asm_comments() {
        add_span_comment(cx, s.span, &format!("{:?}", s));
    }

    let mut bcx = cx;

    let id = ast_util::stmt_id(s);
    let cleanup_debug_loc =
        debuginfo::get_cleanup_debug_loc_for_ast_node(bcx.ccx(), id, s.span, false);
    fcx.push_ast_cleanup_scope(cleanup_debug_loc);

    match s.node {
        hir::StmtExpr(ref e, _) | hir::StmtSemi(ref e, _) => {
            bcx = trans_stmt_semi(bcx, &**e);
        }
        hir::StmtDecl(ref d, _) => {
            match d.node {
                hir::DeclLocal(ref local) => {
                    bcx = init_local(bcx, &**local);
                    debuginfo::create_local_var_metadata(bcx, &**local);
                }
                // Inner items are visited by `trans_item`/`trans_meth`.
                hir::DeclItem(_) => {},
            }
        }
    }

    bcx = fcx.pop_and_trans_ast_cleanup_scope(bcx, ast_util::stmt_id(s));

    return bcx;
}

pub fn trans_stmt_semi<'blk, 'tcx>(cx: Block<'blk, 'tcx>, e: &hir::Expr)
                                   -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_stmt_semi");

    if cx.unreachable.get() {
        return cx;
    }

    let ty = expr_ty(cx, e);
    if cx.fcx.type_needs_drop(ty) {
        expr::trans_to_lvalue(cx, e, "stmt").bcx
    } else {
        expr::trans_into(cx, e, expr::Ignore)
    }
}

pub fn trans_block<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               b: &hir::Block,
                               mut dest: expr::Dest)
                               -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_block");

    if bcx.unreachable.get() {
        return bcx;
    }

    let fcx = bcx.fcx;
    let mut bcx = bcx;

    let cleanup_debug_loc =
        debuginfo::get_cleanup_debug_loc_for_ast_node(bcx.ccx(), b.id, b.span, true);
    fcx.push_ast_cleanup_scope(cleanup_debug_loc);

    for s in &b.stmts {
        bcx = trans_stmt(bcx, &**s);
    }

    if dest != expr::Ignore {
        let block_ty = node_id_type(bcx, b.id);

        if b.expr.is_none() || type_is_zero_size(bcx.ccx(), block_ty) {
            dest = expr::Ignore;
        } else if b.expr.is_some() {
            // If the block has an expression, but that expression isn't reachable,
            // don't save into the destination given, ignore it.
            if let Some(ref cfg) = bcx.fcx.cfg {
                if !cfg.node_is_reachable(b.expr.as_ref().unwrap().id) {
                    dest = expr::Ignore;
                }
            }
        }
    }

    match b.expr {
        Some(ref e) => {
            if !bcx.unreachable.get() {
                bcx = expr::trans_into(bcx, &**e, dest);
            }
        }
        None => {
            assert!(dest == expr::Ignore || bcx.unreachable.get());
        }
    }

    bcx = fcx.pop_and_trans_ast_cleanup_scope(bcx, b.id);

    return bcx;
}

pub fn trans_if<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                            if_id: ast::NodeId,
                            cond: &hir::Expr,
                            thn: &hir::Block,
                            els: Option<&hir::Expr>,
                            dest: expr::Dest)
                            -> Block<'blk, 'tcx> {
    debug!("trans_if(bcx={}, if_id={}, cond={:?}, thn={}, dest={})",
           bcx.to_str(), if_id, cond, thn.id,
           dest.to_string(bcx.ccx()));
    let _icx = push_ctxt("trans_if");

    if bcx.unreachable.get() {
        return bcx;
    }

    let mut bcx = bcx;

    let cond_val = unpack_result!(bcx, expr::trans(bcx, cond).to_llbool());

    // Drop branches that are known to be impossible
    if let Some(cv) = const_to_opt_uint(cond_val) {
        if cv == 1 {
            // if true { .. } [else { .. }]
            bcx = trans_block(bcx, &*thn, dest);
            trans::debuginfo::clear_source_location(bcx.fcx);
        } else {
            if let Some(elexpr) = els {
                bcx = expr::trans_into(bcx, &*elexpr, dest);
                trans::debuginfo::clear_source_location(bcx.fcx);
            }
        }

        return bcx;
    }

    let name = format!("then-block-{}-", thn.id);
    let then_bcx_in = bcx.fcx.new_id_block(&name[..], thn.id);
    let then_bcx_out = trans_block(then_bcx_in, &*thn, dest);
    trans::debuginfo::clear_source_location(bcx.fcx);

    let cond_source_loc = cond.debug_loc();

    let next_bcx;
    match els {
        Some(elexpr) => {
            let else_bcx_in = bcx.fcx.new_id_block("else-block", elexpr.id);
            let else_bcx_out = expr::trans_into(else_bcx_in, &*elexpr, dest);
            next_bcx = bcx.fcx.join_blocks(if_id,
                                           &[then_bcx_out, else_bcx_out]);
            CondBr(bcx, cond_val, then_bcx_in.llbb, else_bcx_in.llbb, cond_source_loc);
        }

        None => {
            next_bcx = bcx.fcx.new_id_block("next-block", if_id);
            Br(then_bcx_out, next_bcx.llbb, DebugLoc::None);
            CondBr(bcx, cond_val, then_bcx_in.llbb, next_bcx.llbb, cond_source_loc);
        }
    }

    // Clear the source location because it is still set to whatever has been translated
    // right before.
    trans::debuginfo::clear_source_location(next_bcx.fcx);

    next_bcx
}

pub fn trans_while<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               loop_expr: &hir::Expr,
                               cond: &hir::Expr,
                               body: &hir::Block)
                               -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_while");

    if bcx.unreachable.get() {
        return bcx;
    }

    let fcx = bcx.fcx;

    //            bcx
    //             |
    //         cond_bcx_in  <--------+
    //             |                 |
    //         cond_bcx_out          |
    //           |      |            |
    //           |    body_bcx_in    |
    // cleanup_blk      |            |
    //    |           body_bcx_out --+
    // next_bcx_in

    let next_bcx_in = fcx.new_id_block("while_exit", loop_expr.id);
    let cond_bcx_in = fcx.new_id_block("while_cond", cond.id);
    let body_bcx_in = fcx.new_id_block("while_body", body.id);

    fcx.push_loop_cleanup_scope(loop_expr.id, [next_bcx_in, cond_bcx_in]);

    Br(bcx, cond_bcx_in.llbb, loop_expr.debug_loc());

    // compile the block where we will handle loop cleanups
    let cleanup_llbb = fcx.normal_exit_block(loop_expr.id, cleanup::EXIT_BREAK);

    // compile the condition
    let Result {bcx: cond_bcx_out, val: cond_val} =
        expr::trans(cond_bcx_in, cond).to_llbool();

    CondBr(cond_bcx_out, cond_val, body_bcx_in.llbb, cleanup_llbb, cond.debug_loc());

    // loop body:
    let body_bcx_out = trans_block(body_bcx_in, body, expr::Ignore);
    Br(body_bcx_out, cond_bcx_in.llbb, DebugLoc::None);

    fcx.pop_loop_cleanup_scope(loop_expr.id);
    return next_bcx_in;
}

pub fn trans_loop<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              loop_expr: &hir::Expr,
                              body: &hir::Block)
                              -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_loop");

    if bcx.unreachable.get() {
        return bcx;
    }

    let fcx = bcx.fcx;

    //            bcx
    //             |
    //         body_bcx_in
    //             |
    //         body_bcx_out
    //
    // next_bcx
    //
    // Links between body_bcx_in and next_bcx are created by
    // break statements.

    let next_bcx_in = bcx.fcx.new_id_block("loop_exit", loop_expr.id);
    let body_bcx_in = bcx.fcx.new_id_block("loop_body", body.id);

    fcx.push_loop_cleanup_scope(loop_expr.id, [next_bcx_in, body_bcx_in]);

    Br(bcx, body_bcx_in.llbb, loop_expr.debug_loc());
    let body_bcx_out = trans_block(body_bcx_in, body, expr::Ignore);
    Br(body_bcx_out, body_bcx_in.llbb, DebugLoc::None);

    fcx.pop_loop_cleanup_scope(loop_expr.id);

    // If there are no predecessors for the next block, we just translated an endless loop and the
    // next block is unreachable
    if BasicBlock(next_bcx_in.llbb).pred_iter().next().is_none() {
        Unreachable(next_bcx_in);
    }

    return next_bcx_in;
}

pub fn trans_break_cont<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                    expr: &hir::Expr,
                                    opt_label: Option<ast::Name>,
                                    exit: usize)
                                    -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_break_cont");

    if bcx.unreachable.get() {
        return bcx;
    }

    let fcx = bcx.fcx;

    // Locate loop that we will break to
    let loop_id = match opt_label {
        None => fcx.top_loop_scope(),
        Some(_) => {
            match bcx.tcx().def_map.borrow().get(&expr.id).map(|d| d.full_def())  {
                Some(def::DefLabel(loop_id)) => loop_id,
                r => {
                    bcx.tcx().sess.bug(&format!("{:?} in def-map for label", r))
                }
            }
        }
    };

    // Generate appropriate cleanup code and branch
    let cleanup_llbb = fcx.normal_exit_block(loop_id, exit);
    Br(bcx, cleanup_llbb, expr.debug_loc());
    Unreachable(bcx); // anything afterwards should be ignored
    return bcx;
}

pub fn trans_break<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               expr: &hir::Expr,
                               label_opt: Option<ast::Name>)
                               -> Block<'blk, 'tcx> {
    return trans_break_cont(bcx, expr, label_opt, cleanup::EXIT_BREAK);
}

pub fn trans_cont<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              expr: &hir::Expr,
                              label_opt: Option<ast::Name>)
                              -> Block<'blk, 'tcx> {
    return trans_break_cont(bcx, expr, label_opt, cleanup::EXIT_LOOP);
}

pub fn trans_ret<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                             return_expr: &hir::Expr,
                             retval_expr: Option<&hir::Expr>)
                             -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_ret");

    if bcx.unreachable.get() {
        return bcx;
    }

    let fcx = bcx.fcx;
    let mut bcx = bcx;
    let dest = match (fcx.llretslotptr.get(), retval_expr) {
        (Some(_), Some(retval_expr)) => {
            let ret_ty = expr_ty_adjusted(bcx, &*retval_expr);
            expr::SaveIn(fcx.get_ret_slot(bcx, ty::FnConverging(ret_ty), "ret_slot"))
        }
        _ => expr::Ignore,
    };
    if let Some(x) = retval_expr {
        bcx = expr::trans_into(bcx, &*x, dest);
        match dest {
            expr::SaveIn(slot) if fcx.needs_ret_allocas => {
                Store(bcx, slot, fcx.llretslotptr.get().unwrap());
            }
            _ => {}
        }
    }
    let cleanup_llbb = fcx.return_exit_block();
    Br(bcx, cleanup_llbb, return_expr.debug_loc());
    Unreachable(bcx);
    return bcx;
}

pub fn trans_fail<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              call_info: NodeIdAndSpan,
                              fail_str: InternedString)
                              -> Block<'blk, 'tcx> {
    let ccx = bcx.ccx();
    let _icx = push_ctxt("trans_fail_value");

    if bcx.unreachable.get() {
        return bcx;
    }

    let v_str = C_str_slice(ccx, fail_str);
    let loc = bcx.sess().codemap().lookup_char_pos(call_info.span.lo);
    let filename = token::intern_and_get_ident(&loc.file.name);
    let filename = C_str_slice(ccx, filename);
    let line = C_u32(ccx, loc.line as u32);
    let expr_file_line_const = C_struct(ccx, &[v_str, filename, line], false);
    let align = machine::llalign_of_min(ccx, val_ty(expr_file_line_const));
    let expr_file_line = consts::addr_of(ccx, expr_file_line_const, align, "panic_loc");
    let args = vec!(expr_file_line);
    let did = langcall(bcx, Some(call_info.span), "", PanicFnLangItem);
    let bcx = callee::trans_lang_call(bcx,
                                      did,
                                      &args[..],
                                      Some(expr::Ignore),
                                      call_info.debug_loc()).bcx;
    Unreachable(bcx);
    return bcx;
}

pub fn trans_fail_bounds_check<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                           call_info: NodeIdAndSpan,
                                           index: ValueRef,
                                           len: ValueRef)
                                           -> Block<'blk, 'tcx> {
    let ccx = bcx.ccx();
    let _icx = push_ctxt("trans_fail_bounds_check");

    if bcx.unreachable.get() {
        return bcx;
    }

    // Extract the file/line from the span
    let loc = bcx.sess().codemap().lookup_char_pos(call_info.span.lo);
    let filename = token::intern_and_get_ident(&loc.file.name);

    // Invoke the lang item
    let filename = C_str_slice(ccx,  filename);
    let line = C_u32(ccx, loc.line as u32);
    let file_line_const = C_struct(ccx, &[filename, line], false);
    let align = machine::llalign_of_min(ccx, val_ty(file_line_const));
    let file_line = consts::addr_of(ccx, file_line_const, align, "panic_bounds_check_loc");
    let args = vec!(file_line, index, len);
    let did = langcall(bcx, Some(call_info.span), "", PanicBoundsCheckFnLangItem);
    let bcx = callee::trans_lang_call(bcx,
                                      did,
                                      &args[..],
                                      Some(expr::Ignore),
                                      call_info.debug_loc()).bcx;
    Unreachable(bcx);
    return bcx;
}
