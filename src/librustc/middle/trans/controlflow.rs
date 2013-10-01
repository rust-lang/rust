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
use middle::trans::expr;
use middle::ty;
use util::common::indenter;
use util::ppaux;

use middle::trans::type_::Type;

use syntax::ast;
use syntax::ast::Name;
use syntax::ast_util;
use syntax::codemap::Span;
use syntax::visit::Visitor;

pub fn trans_block(bcx: @mut Block, b: &ast::Block, dest: expr::Dest) -> @mut Block {
    let _icx = push_ctxt("trans_block");
    let mut bcx = bcx;
    for s in b.stmts.iter() {
        bcx = trans_stmt(bcx, *s);
    }
    match b.expr {
        Some(e) => {
            bcx = expr::trans_into(bcx, e, dest);
        }
        None => {
            assert!(dest == expr::Ignore || bcx.unreachable);
        }
    }
    return bcx;
}

pub fn trans_if(bcx: @mut Block,
            cond: &ast::Expr,
            thn: &ast::Block,
            els: Option<@ast::Expr>,
            dest: expr::Dest)
         -> @mut Block {
    debug2!("trans_if(bcx={}, cond={}, thn={:?}, dest={})",
           bcx.to_str(), bcx.expr_to_str(cond), thn.id,
           dest.to_str(bcx.ccx()));
    let _indenter = indenter();

    let _icx = push_ctxt("trans_if");

    let Result {bcx, val: cond_val} =
        expr::trans_to_datum(bcx, cond).to_result();

    let cond_val = bool_to_i1(bcx, cond_val);

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
            return do with_scope(bcx, thn.info(), "if_true_then") |bcx| {
                let bcx_out = trans_block(bcx, thn, dest);
                trans_block_cleanups(bcx_out, block_cleanups(bcx))
            }
        } else {
            let mut trans = TransItemVisitor { ccx: bcx.fcx.ccx } ;
            trans.visit_block(thn, ());

            match els {
                // if false { .. } else { .. }
                Some(elexpr) => {
                    return do with_scope(bcx, elexpr.info(), "if_false_then") |bcx| {
                        let bcx_out = trans_if_else(bcx, elexpr, dest);
                        trans_block_cleanups(bcx_out, block_cleanups(bcx))
                    }
                }
                // if false { .. }
                None => return bcx,
            }
        }
    }

    let then_bcx_in = scope_block(bcx, thn.info(), "then");

    let then_bcx_out = trans_block(then_bcx_in, thn, dest);
    let then_bcx_out = trans_block_cleanups(then_bcx_out,
                                            block_cleanups(then_bcx_in));

    // Calling trans_block directly instead of trans_expr
    // because trans_expr will create another scope block
    // context for the block, but we've already got the
    // 'else' context
    let (else_bcx_in, next_bcx) = match els {
      Some(elexpr) => {
          let else_bcx_in = scope_block(bcx, elexpr.info(), "else");
          let else_bcx_out = trans_if_else(else_bcx_in, elexpr, dest);
          (else_bcx_in, join_blocks(bcx, [then_bcx_out, else_bcx_out]))
      }
      _ => {
          let next_bcx = sub_block(bcx, "next");
          Br(then_bcx_out, next_bcx.llbb);

          (next_bcx, next_bcx)
      }
    };

    debug2!("then_bcx_in={}, else_bcx_in={}",
           then_bcx_in.to_str(), else_bcx_in.to_str());

    CondBr(bcx, cond_val, then_bcx_in.llbb, else_bcx_in.llbb);
    return next_bcx;

    // trans `else [ if { .. } ... | { .. } ]`
    fn trans_if_else(else_bcx_in: @mut Block, elexpr: @ast::Expr,
                     dest: expr::Dest) -> @mut Block {
        let else_bcx_out = match elexpr.node {
            ast::ExprIf(_, _, _) => {
                let elseif_blk = ast_util::block_from_expr(elexpr);
                trans_block(else_bcx_in, &elseif_blk, dest)
            }
            ast::ExprBlock(ref blk) => {
                trans_block(else_bcx_in, blk, dest)
            }
            // would be nice to have a constraint on ifs
            _ => else_bcx_in.tcx().sess.bug("strange alternative in if")
        };
        trans_block_cleanups(else_bcx_out, block_cleanups(else_bcx_in))
    }
}

pub fn join_blocks(parent_bcx: @mut Block, in_cxs: &[@mut Block]) -> @mut Block {
    let out = sub_block(parent_bcx, "join");
    let mut reachable = false;
    for bcx in in_cxs.iter() {
        if !bcx.unreachable {
            Br(*bcx, out.llbb);
            reachable = true;
        }
    }
    if !reachable {
        Unreachable(out);
    }
    return out;
}

pub fn trans_while(bcx: @mut Block, cond: &ast::Expr, body: &ast::Block) -> @mut Block {
    let _icx = push_ctxt("trans_while");
    let next_bcx = sub_block(bcx, "while next");

    //            bcx
    //             |
    //          loop_bcx
    //             |
    //         cond_bcx_in  <--------+
    //             |                 |
    //         cond_bcx_out          |
    //           |      |            |
    //           |    body_bcx_in    |
    //    +------+      |            |
    //    |           body_bcx_out --+
    // next_bcx

    let loop_bcx = loop_scope_block(bcx, next_bcx, None, "`while`",
                                    body.info());
    let cond_bcx_in = scope_block(loop_bcx, cond.info(), "while loop cond");
    let body_bcx_in = scope_block(loop_bcx, body.info(), "while loop body");
    Br(bcx, loop_bcx.llbb);
    Br(loop_bcx, cond_bcx_in.llbb);

    // compile the condition
    let Result {bcx: cond_bcx_out, val: cond_val} =
        expr::trans_to_datum(cond_bcx_in, cond).to_result();
    let cond_val = bool_to_i1(cond_bcx_out, cond_val);
    let cond_bcx_out =
        trans_block_cleanups(cond_bcx_out, block_cleanups(cond_bcx_in));
    CondBr(cond_bcx_out, cond_val, body_bcx_in.llbb, next_bcx.llbb);

    // loop body:
    let body_bcx_out = trans_block(body_bcx_in, body, expr::Ignore);
    cleanup_and_Br(body_bcx_out, body_bcx_in, cond_bcx_in.llbb);

    return next_bcx;
}

pub fn trans_loop(bcx:@mut Block,
                  body: &ast::Block,
                  opt_label: Option<Name>)
               -> @mut Block {
    let _icx = push_ctxt("trans_loop");
    let next_bcx = sub_block(bcx, "next");
    let body_bcx_in = loop_scope_block(bcx, next_bcx, opt_label, "`loop`",
                                       body.info());
    Br(bcx, body_bcx_in.llbb);
    let body_bcx_out = trans_block(body_bcx_in, body, expr::Ignore);
    cleanup_and_Br(body_bcx_out, body_bcx_in, body_bcx_in.llbb);
    return next_bcx;
}

pub fn trans_break_cont(bcx: @mut Block,
                        opt_label: Option<Name>,
                        to_end: bool)
                     -> @mut Block {
    let _icx = push_ctxt("trans_break_cont");
    // Locate closest loop block, outputting cleanup as we go.
    let mut unwind = bcx;
    let mut cur_scope = unwind.scope;
    let mut target;
    loop {
        cur_scope = match cur_scope {
            Some(@ScopeInfo {
                loop_break: Some(brk),
                loop_label: l,
                parent,
                _
            }) => {
                // If we're looking for a labeled loop, check the label...
                target = if to_end {
                    brk
                } else {
                    unwind
                };
                match opt_label {
                    Some(desired) => match l {
                        Some(actual) if actual == desired => break,
                        // If it doesn't match the one we want,
                        // don't break
                        _ => parent,
                    },
                    None => break,
                }
            }
            Some(inf) => inf.parent,
            None => {
                unwind = match unwind.parent {
                    Some(bcx) => bcx,
                        // This is a return from a loop body block
                        None => {
                            Store(bcx, C_bool(!to_end), bcx.fcx.llretptr.unwrap());
                            cleanup_and_leave(bcx, None, Some(bcx.fcx.get_llreturn()));
                            Unreachable(bcx);
                            return bcx;
                        }
                };
                unwind.scope
            }
        }
    }
    cleanup_and_Br(bcx, unwind, target.llbb);
    Unreachable(bcx);
    return bcx;
}

pub fn trans_break(bcx: @mut Block, label_opt: Option<Name>) -> @mut Block {
    return trans_break_cont(bcx, label_opt, true);
}

pub fn trans_cont(bcx: @mut Block, label_opt: Option<Name>) -> @mut Block {
    return trans_break_cont(bcx, label_opt, false);
}

pub fn trans_ret(bcx: @mut Block, e: Option<@ast::Expr>) -> @mut Block {
    let _icx = push_ctxt("trans_ret");
    let mut bcx = bcx;
    let dest = match bcx.fcx.llretptr {
        None => expr::Ignore,
        Some(retptr) => expr::SaveIn(retptr),
    };
    match e {
        Some(x) => {
            bcx = expr::trans_into(bcx, x, dest);
        }
        _ => ()
    }
    cleanup_and_leave(bcx, None, Some(bcx.fcx.get_llreturn()));
    Unreachable(bcx);
    return bcx;
}

pub fn trans_fail_expr(bcx: @mut Block,
                       sp_opt: Option<Span>,
                       fail_expr: Option<@ast::Expr>)
                    -> @mut Block {
    let _icx = push_ctxt("trans_fail_expr");
    let mut bcx = bcx;
    match fail_expr {
        Some(arg_expr) => {
            let ccx = bcx.ccx();
            let tcx = ccx.tcx;
            let arg_datum = unpack_datum!(
                bcx, expr::trans_to_datum(bcx, arg_expr));

            if ty::type_is_str(arg_datum.ty) {
                let (lldata, _) = arg_datum.get_vec_base_and_len_no_root(bcx);
                return trans_fail_value(bcx, sp_opt, lldata);
            } else if bcx.unreachable || ty::type_is_bot(arg_datum.ty) {
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

pub fn trans_fail(bcx: @mut Block,
                  sp_opt: Option<Span>,
                  fail_str: @str)
               -> @mut Block {
    let _icx = push_ctxt("trans_fail");
    let V_fail_str = C_cstr(bcx.ccx(), fail_str);
    return trans_fail_value(bcx, sp_opt, V_fail_str);
}

fn trans_fail_value(bcx: @mut Block,
                    sp_opt: Option<Span>,
                    V_fail_str: ValueRef)
                 -> @mut Block {
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

pub fn trans_fail_bounds_check(bcx: @mut Block, sp: Span,
                               index: ValueRef, len: ValueRef) -> @mut Block {
    let _icx = push_ctxt("trans_fail_bounds_check");
    let (filename, line) = filename_and_line_num_from_span(bcx, sp);
    let args = ~[filename, line, index, len];
    let did = langcall(bcx, Some(sp), "", FailBoundsCheckFnLangItem);
    let bcx = callee::trans_lang_call(bcx, did, args, Some(expr::Ignore)).bcx;
    Unreachable(bcx);
    return bcx;
}
