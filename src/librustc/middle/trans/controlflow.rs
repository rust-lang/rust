// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lib::llvm::*;
use middle::lang_items::{FailFnLangItem, FailBoundsCheckFnLangItem};
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::common::*;
use middle::trans::debuginfo;
use middle::trans::cleanup;
use middle::trans::cleanup::CleanupMethods;
use middle::trans::expr;
use middle::ty;
use util::ppaux;
use util::ppaux::Repr;

use middle::trans::type_::Type;

use syntax::ast;
use syntax::ast::Name;
use syntax::ast_util;
use syntax::codemap::Span;
use syntax::visit::Visitor;

pub fn trans_stmt<'a>(cx: &'a Block<'a>,
                      s: &ast::Stmt)
                      -> &'a Block<'a> {
    let _icx = push_ctxt("trans_stmt");
    let fcx = cx.fcx;
    debug!("trans_stmt({})", s.repr(cx.tcx()));

    if cx.sess().asm_comments() {
        add_span_comment(cx, s.span, s.repr(cx.tcx()));
    }

    let mut bcx = cx;

    let id = ast_util::stmt_id(s);
    fcx.push_ast_cleanup_scope(id);

    match s.node {
        ast::StmtExpr(e, _) | ast::StmtSemi(e, _) => {
            bcx = expr::trans_into(cx, e, expr::Ignore);
        }
        ast::StmtDecl(d, _) => {
            match d.node {
                ast::DeclLocal(ref local) => {
                    bcx = init_local(bcx, *local);
                    if cx.sess().opts.extra_debuginfo {
                        debuginfo::create_local_var_metadata(bcx, *local);
                    }
                }
                ast::DeclItem(i) => trans_item(cx.fcx.ccx, i)
            }
        }
        ast::StmtMac(..) => cx.tcx().sess.bug("unexpanded macro")
    }

    bcx = fcx.pop_and_trans_ast_cleanup_scope(
        bcx, ast_util::stmt_id(s));

    return bcx;
}

pub fn trans_block<'a>(bcx: &'a Block<'a>,
                       b: &ast::Block,
                       dest: expr::Dest)
                       -> &'a Block<'a> {
    let _icx = push_ctxt("trans_block");
    let fcx = bcx.fcx;
    let mut bcx = bcx;

    fcx.push_ast_cleanup_scope(b.id);

    for s in b.stmts.iter() {
        bcx = trans_stmt(bcx, *s);
    }
    match b.expr {
        Some(e) => {
            bcx = expr::trans_into(bcx, e, dest);
        }
        None => {
            assert!(dest == expr::Ignore || bcx.unreachable.get());
        }
    }

    bcx = fcx.pop_and_trans_ast_cleanup_scope(bcx, b.id);

    return bcx;
}

pub fn trans_if<'a>(bcx: &'a Block<'a>,
                    if_id: ast::NodeId,
                    cond: &ast::Expr,
                    thn: ast::P<ast::Block>,
                    els: Option<@ast::Expr>,
                    dest: expr::Dest)
                    -> &'a Block<'a> {
    debug!("trans_if(bcx={}, if_id={}, cond={}, thn={:?}, dest={})",
           bcx.to_str(), if_id, bcx.expr_to_str(cond), thn.id,
           dest.to_str(bcx.ccx()));
    let _icx = push_ctxt("trans_if");
    let mut bcx = bcx;

    let cond_val = unpack_result!(bcx, expr::trans(bcx, cond).to_llbool());

    // Drop branches that are known to be impossible
    if is_const(cond_val) && !is_undef(cond_val) {
        if const_to_uint(cond_val) == 1 {
            match els {
                Some(elexpr) => {
                    let mut trans = TransItemVisitor { ccx: bcx.fcx.ccx };
                    trans.visit_expr(elexpr, ());
                }
                None => {}
            }
            // if true { .. } [else { .. }]
            bcx = trans_block(bcx, thn, dest);
            debuginfo::clear_source_location(bcx.fcx);
        } else {
            let mut trans = TransItemVisitor { ccx: bcx.fcx.ccx } ;
            trans.visit_block(thn, ());

            match els {
                // if false { .. } else { .. }
                Some(elexpr) => {
                    bcx = expr::trans_into(bcx, elexpr, dest);
                    debuginfo::clear_source_location(bcx.fcx);
                }

                // if false { .. }
                None => { }
            }
        }

        return bcx;
    }

    let name = format!("then-block-{}-", thn.id);
    let then_bcx_in = bcx.fcx.new_id_block(name, thn.id);
    let then_bcx_out = trans_block(then_bcx_in, thn, dest);
    debuginfo::clear_source_location(bcx.fcx);

    let next_bcx;
    match els {
        Some(elexpr) => {
            let else_bcx_in = bcx.fcx.new_id_block("else-block", elexpr.id);
            let else_bcx_out = expr::trans_into(else_bcx_in, elexpr, dest);
            next_bcx = bcx.fcx.join_blocks(if_id,
                                           [then_bcx_out, else_bcx_out]);
            CondBr(bcx, cond_val, then_bcx_in.llbb, else_bcx_in.llbb);
        }

        None => {
            next_bcx = bcx.fcx.new_id_block("next-block", if_id);
            Br(then_bcx_out, next_bcx.llbb);
            CondBr(bcx, cond_val, then_bcx_in.llbb, next_bcx.llbb);
        }
    }

    // Clear the source location because it is still set to whatever has been translated
    // right before.
    debuginfo::clear_source_location(next_bcx.fcx);

    next_bcx
}

pub fn trans_while<'a>(bcx: &'a Block<'a>,
                       loop_id: ast::NodeId,
                       cond: &ast::Expr,
                       body: &ast::Block)
                       -> &'a Block<'a> {
    let _icx = push_ctxt("trans_while");
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

    let next_bcx_in = fcx.new_id_block("while_exit", loop_id);
    let cond_bcx_in = fcx.new_id_block("while_cond", cond.id);
    let body_bcx_in = fcx.new_id_block("while_body", body.id);

    fcx.push_loop_cleanup_scope(loop_id, [next_bcx_in, cond_bcx_in]);

    Br(bcx, cond_bcx_in.llbb);

    // compile the block where we will handle loop cleanups
    let cleanup_llbb = fcx.normal_exit_block(loop_id, cleanup::EXIT_BREAK);

    // compile the condition
    let Result {bcx: cond_bcx_out, val: cond_val} =
        expr::trans(cond_bcx_in, cond).to_llbool();
    CondBr(cond_bcx_out, cond_val, body_bcx_in.llbb, cleanup_llbb);

    // loop body:
    let body_bcx_out = trans_block(body_bcx_in, body, expr::Ignore);
    Br(body_bcx_out, cond_bcx_in.llbb);

    fcx.pop_loop_cleanup_scope(loop_id);
    return next_bcx_in;
}

pub fn trans_loop<'a>(bcx:&'a Block<'a>,
                      loop_id: ast::NodeId,
                      body: &ast::Block)
                      -> &'a Block<'a> {
    let _icx = push_ctxt("trans_loop");
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

    let next_bcx_in = bcx.fcx.new_id_block("loop_exit", loop_id);
    let body_bcx_in = bcx.fcx.new_id_block("loop_body", body.id);

    fcx.push_loop_cleanup_scope(loop_id, [next_bcx_in, body_bcx_in]);

    Br(bcx, body_bcx_in.llbb);
    let body_bcx_out = trans_block(body_bcx_in, body, expr::Ignore);
    Br(body_bcx_out, body_bcx_in.llbb);

    fcx.pop_loop_cleanup_scope(loop_id);

    return next_bcx_in;
}

pub fn trans_break_cont<'a>(bcx: &'a Block<'a>,
                            expr_id: ast::NodeId,
                            opt_label: Option<Name>,
                            exit: uint)
                            -> &'a Block<'a> {
    let _icx = push_ctxt("trans_break_cont");
    let fcx = bcx.fcx;

    if bcx.unreachable.get() {
        return bcx;
    }

    // Locate loop that we will break to
    let loop_id = match opt_label {
        None => fcx.top_loop_scope(),
        Some(_) => {
            let def_map = bcx.tcx().def_map.borrow();
            match def_map.get().find(&expr_id) {
                Some(&ast::DefLabel(loop_id)) => loop_id,
                ref r => {
                    bcx.tcx().sess.bug(format!("{:?} in def-map for label", r))
                }
            }
        }
    };

    // Generate appropriate cleanup code and branch
    let cleanup_llbb = fcx.normal_exit_block(loop_id, exit);
    Br(bcx, cleanup_llbb);
    Unreachable(bcx); // anything afterwards should be ignored
    return bcx;
}

pub fn trans_break<'a>(bcx: &'a Block<'a>,
                       expr_id: ast::NodeId,
                       label_opt: Option<Name>)
                       -> &'a Block<'a> {
    return trans_break_cont(bcx, expr_id, label_opt, cleanup::EXIT_BREAK);
}

pub fn trans_cont<'a>(bcx: &'a Block<'a>,
                      expr_id: ast::NodeId,
                      label_opt: Option<Name>)
                      -> &'a Block<'a> {
    return trans_break_cont(bcx, expr_id, label_opt, cleanup::EXIT_LOOP);
}

pub fn trans_ret<'a>(bcx: &'a Block<'a>,
                     e: Option<@ast::Expr>)
                     -> &'a Block<'a> {
    let _icx = push_ctxt("trans_ret");
    let fcx = bcx.fcx;
    let mut bcx = bcx;
    let dest = match bcx.fcx.llretptr.get() {
        None => expr::Ignore,
        Some(retptr) => expr::SaveIn(retptr),
    };
    match e {
        Some(x) => {
            bcx = expr::trans_into(bcx, x, dest);
        }
        _ => ()
    }
    let cleanup_llbb = fcx.return_exit_block();
    Br(bcx, cleanup_llbb);
    Unreachable(bcx);
    return bcx;
}

pub fn trans_fail_expr<'a>(
                       bcx: &'a Block<'a>,
                       sp_opt: Option<Span>,
                       fail_expr: Option<@ast::Expr>)
                       -> &'a Block<'a> {
    let _icx = push_ctxt("trans_fail_expr");
    let mut bcx = bcx;
    match fail_expr {
        Some(arg_expr) => {
            let ccx = bcx.ccx();
            let tcx = ccx.tcx;
            let arg_datum =
                unpack_datum!(bcx, expr::trans_to_lvalue(bcx, arg_expr, "fail"));

            if ty::type_is_str(arg_datum.ty) {
                let (lldata, _) = arg_datum.get_vec_base_and_len_no_root(bcx);
                return trans_fail_value(bcx, sp_opt, lldata);
            } else if bcx.unreachable.get() || ty::type_is_bot(arg_datum.ty) {
                return bcx;
            } else {
                bcx.sess().span_bug(
                    arg_expr.span, ~"fail called with unsupported type " +
                    ppaux::ty_to_str(tcx, arg_datum.ty));
            }
        }
        _ => trans_fail(bcx, sp_opt, @"explicit failure")
    }
}

pub fn trans_fail<'a>(
                  bcx: &'a Block<'a>,
                  sp_opt: Option<Span>,
                  fail_str: @str)
                  -> &'a Block<'a> {
    let _icx = push_ctxt("trans_fail");
    let V_fail_str = C_cstr(bcx.ccx(), fail_str);
    return trans_fail_value(bcx, sp_opt, V_fail_str);
}

fn trans_fail_value<'a>(
                    bcx: &'a Block<'a>,
                    sp_opt: Option<Span>,
                    V_fail_str: ValueRef)
                    -> &'a Block<'a> {
    let _icx = push_ctxt("trans_fail_value");
    let ccx = bcx.ccx();
    let (V_filename, V_line) = match sp_opt {
      Some(sp) => {
        let sess = bcx.sess();
        let loc = sess.parse_sess.cm.lookup_char_pos(sp.lo);
        (C_cstr(bcx.ccx(), loc.file.name),
         loc.line as int)
      }
      None => {
        (C_cstr(bcx.ccx(), @"<runtime>"), 0)
      }
    };
    let V_str = PointerCast(bcx, V_fail_str, Type::i8p());
    let V_filename = PointerCast(bcx, V_filename, Type::i8p());
    let args = ~[V_str, V_filename, C_int(ccx, V_line)];
    let did = langcall(bcx, sp_opt, "", FailFnLangItem);
    let bcx = callee::trans_lang_call(bcx, did, args, Some(expr::Ignore)).bcx;
    Unreachable(bcx);
    return bcx;
}

pub fn trans_fail_bounds_check<'a>(
                               bcx: &'a Block<'a>,
                               sp: Span,
                               index: ValueRef,
                               len: ValueRef)
                               -> &'a Block<'a> {
    let _icx = push_ctxt("trans_fail_bounds_check");
    let (filename, line) = filename_and_line_num_from_span(bcx, sp);
    let args = ~[filename, line, index, len];
    let did = langcall(bcx, Some(sp), "", FailBoundsCheckFnLangItem);
    let bcx = callee::trans_lang_call(bcx, did, args, Some(expr::Ignore)).bcx;
    Unreachable(bcx);
    return bcx;
}
