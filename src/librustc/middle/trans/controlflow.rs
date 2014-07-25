// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::*;
use driver::config::FullDebugInfo;
use middle::def;
use middle::lang_items::{FailFnLangItem, FailBoundsCheckFnLangItem};
use middle::trans::_match;
use middle::trans::adt;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::cleanup::CleanupMethods;
use middle::trans::cleanup;
use middle::trans::common::*;
use middle::trans::datum;
use middle::trans::debuginfo;
use middle::trans::expr;
use middle::trans::meth;
use middle::trans::type_::Type;
use middle::ty;
use middle::typeck::MethodCall;
use util::ppaux::Repr;
use util::ppaux;

use syntax::ast;
use syntax::ast::Ident;
use syntax::ast_util;
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use syntax::parse::token;
use syntax::visit::Visitor;

use std::gc::Gc;

pub fn trans_stmt<'a>(cx: &'a Block<'a>,
                      s: &ast::Stmt)
                      -> &'a Block<'a> {
    let _icx = push_ctxt("trans_stmt");
    let fcx = cx.fcx;
    debug!("trans_stmt({})", s.repr(cx.tcx()));

    if cx.sess().asm_comments() {
        add_span_comment(cx, s.span, s.repr(cx.tcx()).as_slice());
    }

    let mut bcx = cx;

    let id = ast_util::stmt_id(s);
    fcx.push_ast_cleanup_scope(id);

    match s.node {
        ast::StmtExpr(ref e, _) | ast::StmtSemi(ref e, _) => {
            bcx = trans_stmt_semi(bcx, &**e);
        }
        ast::StmtDecl(d, _) => {
            match d.node {
                ast::DeclLocal(ref local) => {
                    bcx = init_local(bcx, &**local);
                    if cx.sess().opts.debuginfo == FullDebugInfo {
                        debuginfo::create_local_var_metadata(bcx, &**local);
                    }
                }
                ast::DeclItem(ref i) => trans_item(cx.fcx.ccx, &**i)
            }
        }
        ast::StmtMac(..) => cx.tcx().sess.bug("unexpanded macro")
    }

    bcx = fcx.pop_and_trans_ast_cleanup_scope(
        bcx, ast_util::stmt_id(s));

    return bcx;
}

pub fn trans_stmt_semi<'a>(cx: &'a Block<'a>, e: &ast::Expr) -> &'a Block<'a> {
    let _icx = push_ctxt("trans_stmt_semi");
    let ty = expr_ty(cx, e);
    if ty::type_needs_drop(cx.tcx(), ty) {
        expr::trans_to_lvalue(cx, e, "stmt").bcx
    } else {
        expr::trans_into(cx, e, expr::Ignore)
    }
}

pub fn trans_block<'a>(bcx: &'a Block<'a>,
                       b: &ast::Block,
                       mut dest: expr::Dest)
                       -> &'a Block<'a> {
    let _icx = push_ctxt("trans_block");
    let fcx = bcx.fcx;
    let mut bcx = bcx;

    fcx.push_ast_cleanup_scope(b.id);

    for s in b.stmts.iter() {
        bcx = trans_stmt(bcx, &**s);
    }

    if dest != expr::Ignore {
        let block_ty = node_id_type(bcx, b.id);
        if b.expr.is_none() || type_is_zero_size(bcx.ccx(), block_ty) {
            dest = expr::Ignore;
        }
    }

    match b.expr {
        Some(ref e) => {
            bcx = expr::trans_into(bcx, &**e, dest);
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
                    els: Option<Gc<ast::Expr>>,
                    dest: expr::Dest)
                    -> &'a Block<'a> {
    debug!("trans_if(bcx={}, if_id={}, cond={}, thn={:?}, dest={})",
           bcx.to_str(), if_id, bcx.expr_to_string(cond), thn.id,
           dest.to_string(bcx.ccx()));
    let _icx = push_ctxt("trans_if");
    let mut bcx = bcx;

    let cond_val = unpack_result!(bcx, expr::trans(bcx, cond).to_llbool());

    // Drop branches that are known to be impossible
    if is_const(cond_val) && !is_undef(cond_val) {
        if const_to_uint(cond_val) == 1 {
            match els {
                Some(elexpr) => {
                    let mut trans = TransItemVisitor { ccx: bcx.fcx.ccx };
                    trans.visit_expr(&*elexpr, ());
                }
                None => {}
            }
            // if true { .. } [else { .. }]
            bcx = trans_block(bcx, &*thn, dest);
            debuginfo::clear_source_location(bcx.fcx);
        } else {
            let mut trans = TransItemVisitor { ccx: bcx.fcx.ccx } ;
            trans.visit_block(&*thn, ());

            match els {
                // if false { .. } else { .. }
                Some(elexpr) => {
                    bcx = expr::trans_into(bcx, &*elexpr, dest);
                    debuginfo::clear_source_location(bcx.fcx);
                }

                // if false { .. }
                None => { }
            }
        }

        return bcx;
    }

    let name = format!("then-block-{}-", thn.id);
    let then_bcx_in = bcx.fcx.new_id_block(name.as_slice(), thn.id);
    let then_bcx_out = trans_block(then_bcx_in, &*thn, dest);
    debuginfo::clear_source_location(bcx.fcx);

    let next_bcx;
    match els {
        Some(elexpr) => {
            let else_bcx_in = bcx.fcx.new_id_block("else-block", elexpr.id);
            let else_bcx_out = expr::trans_into(else_bcx_in, &*elexpr, dest);
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

/// Translates a `for` loop.
pub fn trans_for<'a>(
                 mut bcx: &'a Block<'a>,
                 loop_info: NodeInfo,
                 pat: Gc<ast::Pat>,
                 head: &ast::Expr,
                 body: &ast::Block)
                 -> &'a Block<'a> {
    let _icx = push_ctxt("trans_for");

    //            bcx
    //             |
    //      loopback_bcx_in  <-------+
    //             |                 |
    //      loopback_bcx_out         |
    //           |      |            |
    //           |    body_bcx_in    |
    // cleanup_blk      |            |
    //    |           body_bcx_out --+
    // next_bcx_in

    // Codegen the head to create the iterator value.
    let iterator_datum =
        unpack_datum!(bcx, expr::trans_to_lvalue(bcx, head, "for_head"));
    let iterator_type = node_id_type(bcx, head.id);
    debug!("iterator type is {}, datum type is {}",
           ppaux::ty_to_string(bcx.tcx(), iterator_type),
           ppaux::ty_to_string(bcx.tcx(), iterator_datum.ty));
    let lliterator = load_ty(bcx, iterator_datum.val, iterator_datum.ty);

    // Create our basic blocks and set up our loop cleanups.
    let next_bcx_in = bcx.fcx.new_id_block("for_exit", loop_info.id);
    let loopback_bcx_in = bcx.fcx.new_id_block("for_loopback", head.id);
    let body_bcx_in = bcx.fcx.new_id_block("for_body", body.id);
    bcx.fcx.push_loop_cleanup_scope(loop_info.id,
                                    [next_bcx_in, loopback_bcx_in]);
    Br(bcx, loopback_bcx_in.llbb);
    let cleanup_llbb = bcx.fcx.normal_exit_block(loop_info.id,
                                                 cleanup::EXIT_BREAK);

    // Set up the method call (to `.next()`).
    let method_call = MethodCall::expr(loop_info.id);
    let method_type = loopback_bcx_in.tcx()
                                     .method_map
                                     .borrow()
                                     .get(&method_call)
                                     .ty;
    let method_type = monomorphize_type(loopback_bcx_in, method_type);
    let method_result_type = ty::ty_fn_ret(method_type);
    let option_cleanup_scope = body_bcx_in.fcx.push_custom_cleanup_scope();
    let option_cleanup_scope_id = cleanup::CustomScope(option_cleanup_scope);

    // Compile the method call (to `.next()`).
    let mut loopback_bcx_out = loopback_bcx_in;
    let option_datum =
        unpack_datum!(loopback_bcx_out,
                      datum::lvalue_scratch_datum(loopback_bcx_out,
                                                  method_result_type,
                                                  "loop_option",
                                                  false,
                                                  option_cleanup_scope_id,
                                                  (),
                                                  |(), bcx, lloption| {
        let Result {
            bcx: bcx,
            val: _
        } = callee::trans_call_inner(bcx,
                                     Some(loop_info),
                                     method_type,
                                     |bcx, arg_cleanup_scope| {
                                         meth::trans_method_callee(
                                             bcx,
                                             method_call,
                                             None,
                                             arg_cleanup_scope)
                                     },
                                     callee::ArgVals([lliterator]),
                                     Some(expr::SaveIn(lloption)));
        bcx
    }));

    // Check the discriminant; if the `None` case, exit the loop.
    let option_representation = adt::represent_type(loopback_bcx_out.ccx(),
                                                    method_result_type);
    let i8_type = Type::i8(loopback_bcx_out.ccx());
    let lldiscriminant = adt::trans_get_discr(loopback_bcx_out,
                                              &*option_representation,
                                              option_datum.val,
                                              Some(i8_type));
    let llzero = C_u8(loopback_bcx_out.ccx(), 0);
    let llcondition = ICmp(loopback_bcx_out, IntNE, lldiscriminant, llzero);
    CondBr(loopback_bcx_out, llcondition, body_bcx_in.llbb, cleanup_llbb);

    // Now we're in the body. Unpack the `Option` value into the programmer-
    // supplied pattern.
    let llpayload = adt::trans_field_ptr(body_bcx_in,
                                         &*option_representation,
                                         option_datum.val,
                                         1,
                                         0);
    let binding_cleanup_scope = body_bcx_in.fcx.push_custom_cleanup_scope();
    let binding_cleanup_scope_id =
        cleanup::CustomScope(binding_cleanup_scope);
    let mut body_bcx_out =
        _match::store_for_loop_binding(body_bcx_in,
                                       pat,
                                       llpayload,
                                       binding_cleanup_scope_id);

    // Codegen the body.
    body_bcx_out = trans_block(body_bcx_out, body, expr::Ignore);
    body_bcx_out.fcx.pop_custom_cleanup_scope(binding_cleanup_scope);
    body_bcx_out =
        body_bcx_out.fcx
                    .pop_and_trans_custom_cleanup_scope(body_bcx_out,
                                                        option_cleanup_scope);
    Br(body_bcx_out, loopback_bcx_in.llbb);

    // Codegen cleanups and leave.
    next_bcx_in.fcx.pop_loop_cleanup_scope(loop_info.id);
    next_bcx_in
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

    if ty::type_is_bot(node_id_type(bcx, loop_id)) {
        Unreachable(next_bcx_in);
    }

    return next_bcx_in;
}

pub fn trans_break_cont<'a>(bcx: &'a Block<'a>,
                            expr_id: ast::NodeId,
                            opt_label: Option<Ident>,
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
            match bcx.tcx().def_map.borrow().find(&expr_id) {
                Some(&def::DefLabel(loop_id)) => loop_id,
                ref r => {
                    bcx.tcx().sess.bug(format!("{:?} in def-map for label",
                                               r).as_slice())
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
                       label_opt: Option<Ident>)
                       -> &'a Block<'a> {
    return trans_break_cont(bcx, expr_id, label_opt, cleanup::EXIT_BREAK);
}

pub fn trans_cont<'a>(bcx: &'a Block<'a>,
                      expr_id: ast::NodeId,
                      label_opt: Option<Ident>)
                      -> &'a Block<'a> {
    return trans_break_cont(bcx, expr_id, label_opt, cleanup::EXIT_LOOP);
}

pub fn trans_ret<'a>(bcx: &'a Block<'a>,
                     e: Option<Gc<ast::Expr>>)
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
            bcx = expr::trans_into(bcx, &*x, dest);
        }
        _ => {}
    }
    let cleanup_llbb = fcx.return_exit_block();
    Br(bcx, cleanup_llbb);
    Unreachable(bcx);
    return bcx;
}

fn str_slice_arg<'a>(bcx: &'a Block<'a>, s: InternedString) -> ValueRef {
    let ccx = bcx.ccx();
    let s = C_str_slice(ccx, s);
    let slot = alloca(bcx, val_ty(s), "__temp");
    Store(bcx, s, slot);
    slot
}

pub fn trans_fail<'a>(
                  bcx: &'a Block<'a>,
                  sp: Span,
                  fail_str: InternedString)
                  -> &'a Block<'a> {
    let ccx = bcx.ccx();
    let _icx = push_ctxt("trans_fail_value");

    let v_str = str_slice_arg(bcx, fail_str);
    let loc = bcx.sess().codemap().lookup_char_pos(sp.lo);
    let filename = token::intern_and_get_ident(loc.file.name.as_slice());
    let v_filename = str_slice_arg(bcx, filename);
    let v_line = loc.line as int;
    let args = vec!(v_str, v_filename, C_int(ccx, v_line));
    let did = langcall(bcx, Some(sp), "", FailFnLangItem);
    let bcx = callee::trans_lang_call(bcx,
                                      did,
                                      args.as_slice(),
                                      Some(expr::Ignore)).bcx;
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

    // Extract the file/line from the span
    let loc = bcx.sess().codemap().lookup_char_pos(sp.lo);
    let filename = token::intern_and_get_ident(loc.file.name.as_slice());

    // Invoke the lang item
    let filename = str_slice_arg(bcx, filename);
    let line = C_int(bcx.ccx(), loc.line as int);
    let args = vec!(filename, line, index, len);
    let did = langcall(bcx, Some(sp), "", FailBoundsCheckFnLangItem);
    let bcx = callee::trans_lang_call(bcx,
                                      did,
                                      args.as_slice(),
                                      Some(expr::Ignore)).bcx;
    Unreachable(bcx);
    return bcx;
}
