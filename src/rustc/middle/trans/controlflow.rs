use lib::llvm::ValueRef;
use common::*;
use datum::*;
use base::*;

fn macros() { include!("macros.rs"); } // FIXME(#3114): Macro import/export.

fn trans_block(bcx: block, b: ast::blk, dest: expr::Dest) -> block {
    let _icx = bcx.insn_ctxt("trans_block");
    let mut bcx = bcx;
    do block_locals(b) |local| {
        bcx = alloc_local(bcx, local);
    };
    for vec::each(b.node.stmts) |s| {
        debuginfo::update_source_pos(bcx, b.span);
        bcx = trans_stmt(bcx, **s);
    }
    match b.node.expr {
        Some(e) => {
            debuginfo::update_source_pos(bcx, e.span);
            bcx = expr::trans_into(bcx, e, dest);
        }
        None => {
            assert dest == expr::Ignore || bcx.unreachable;
        }
    }
    return bcx;
}

fn trans_if(bcx: block,
            cond: @ast::expr,
            thn: ast::blk,
            els: Option<@ast::expr>,
            dest: expr::Dest)
    -> block
{
    debug!("trans_if(bcx=%s, cond=%s, thn=%?, dest=%s)",
           bcx.to_str(), bcx.expr_to_str(cond), thn.node.id,
           dest.to_str(bcx.ccx()));
    let _indenter = indenter();

    let _icx = bcx.insn_ctxt("trans_if");
    let Result {bcx, val: cond_val} =
        expr::trans_to_datum(bcx, cond).to_result();

    let then_bcx_in = scope_block(bcx, thn.info(), ~"then");
    let else_bcx_in = scope_block(bcx, els.info(), ~"else");
    CondBr(bcx, cond_val, then_bcx_in.llbb, else_bcx_in.llbb);

    debug!("then_bcx_in=%s, else_bcx_in=%s",
           then_bcx_in.to_str(), else_bcx_in.to_str());

    let then_bcx_out = trans_block(then_bcx_in, thn, dest);
    let then_bcx_out = trans_block_cleanups(then_bcx_out,
                                            block_cleanups(then_bcx_in));

    // Calling trans_block directly instead of trans_expr
    // because trans_expr will create another scope block
    // context for the block, but we've already got the
    // 'else' context
    let else_bcx_out = match els {
      Some(elexpr) => {
        match elexpr.node {
          ast::expr_if(_, _, _) => {
            let elseif_blk = ast_util::block_from_expr(elexpr);
            trans_block(else_bcx_in, elseif_blk, dest)
          }
          ast::expr_block(blk) => {
            trans_block(else_bcx_in, blk, dest)
          }
          // would be nice to have a constraint on ifs
          _ => bcx.tcx().sess.bug(~"strange alternative in if")
        }
      }
      _ => else_bcx_in
    };
    let else_bcx_out = trans_block_cleanups(else_bcx_out,
                                            block_cleanups(else_bcx_in));
    return join_blocks(bcx, ~[then_bcx_out, else_bcx_out]);

}

fn join_blocks(parent_bcx: block, in_cxs: ~[block]) -> block {
    let out = sub_block(parent_bcx, ~"join");
    let mut reachable = false;
    for vec::each(in_cxs) |bcx| {
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

fn trans_while(bcx: block, cond: @ast::expr, body: ast::blk)
    -> block {
    let _icx = bcx.insn_ctxt("trans_while");
    let next_bcx = sub_block(bcx, ~"while next");

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

    let loop_bcx = loop_scope_block(bcx, next_bcx, ~"`while`", body.info());
    let cond_bcx_in = scope_block(loop_bcx, cond.info(), ~"while loop cond");
    let body_bcx_in = scope_block(loop_bcx, body.info(), ~"while loop body");
    Br(bcx, loop_bcx.llbb);
    Br(loop_bcx, cond_bcx_in.llbb);

    // compile the condition
    let Result {bcx: cond_bcx_out, val: cond_val} =
        expr::trans_to_datum(cond_bcx_in, cond).to_result();
    let cond_bcx_out =
        trans_block_cleanups(cond_bcx_out, block_cleanups(cond_bcx_in));
    CondBr(cond_bcx_out, cond_val, body_bcx_in.llbb, next_bcx.llbb);

    // loop body:
    let body_bcx_out = trans_block(body_bcx_in, body, expr::Ignore);
    cleanup_and_Br(body_bcx_out, body_bcx_in, cond_bcx_in.llbb);

    return next_bcx;
}

fn trans_loop(bcx:block, body: ast::blk) -> block {
    let _icx = bcx.insn_ctxt("trans_loop");
    let next_bcx = sub_block(bcx, ~"next");
    let body_bcx_in = loop_scope_block(bcx, next_bcx, ~"`loop`", body.info());
    Br(bcx, body_bcx_in.llbb);
    let body_bcx_out = trans_block(body_bcx_in, body, expr::Ignore);
    cleanup_and_Br(body_bcx_out, body_bcx_in, body_bcx_in.llbb);
    return next_bcx;
}

fn trans_log(log_ex: @ast::expr,
             lvl: @ast::expr,
             bcx: block,
             e: @ast::expr) -> block
{
    let _icx = bcx.insn_ctxt("trans_log");
    let ccx = bcx.ccx();
    let mut bcx = bcx;
    if ty::type_is_bot(expr_ty(bcx, lvl)) {
       return expr::trans_into(bcx, lvl, expr::Ignore);
    }

    let modpath = vec::append(
        ~[path_mod(ccx.sess.ident_of(ccx.link_meta.name))],
        vec::filter(bcx.fcx.path, |e|
            match e { path_mod(_) => true, _ => false }
        ));
    let modname = path_str(ccx.sess, modpath);

    let global = if ccx.module_data.contains_key(modname) {
        ccx.module_data.get(modname)
    } else {
        let s = link::mangle_internal_name_by_path_and_seq(
            ccx, modpath, ~"loglevel");
        let global = str::as_c_str(s, |buf| {
            llvm::LLVMAddGlobal(ccx.llmod, T_i32(), buf)
        });
        llvm::LLVMSetGlobalConstant(global, False);
        llvm::LLVMSetInitializer(global, C_null(T_i32()));
        lib::llvm::SetLinkage(global, lib::llvm::InternalLinkage);
        ccx.module_data.insert(modname, global);
        global
    };
    let current_level = Load(bcx, global);
    let level = unpack_result!(bcx, {
        do with_scope_result(bcx, lvl.info(), ~"level") |bcx| {
            expr::trans_to_datum(bcx, lvl).to_result()
        }
    });

    let llenabled = ICmp(bcx, lib::llvm::IntUGE, current_level, level);
    do with_cond(bcx, llenabled) |bcx| {
        do with_scope(bcx, log_ex.info(), ~"log") |bcx| {
            let mut bcx = bcx;

            // Translate the value to be logged
            let val_datum = unpack_datum!(bcx, expr::trans_to_datum(bcx, e));

            // Call the polymorphic log function
            let val = val_datum.to_ref_llval(bcx);
            let did = bcx.tcx().lang_items.log_type_fn.get();
            let bcx = callee::trans_rtcall_or_lang_call_with_type_params(
                bcx, did, ~[level, val], ~[val_datum.ty], expr::Ignore);
            bcx
        }
    }
}

fn trans_break_cont(bcx: block, to_end: bool)
    -> block {
    let _icx = bcx.insn_ctxt("trans_break_cont");
    // Locate closest loop block, outputting cleanup as we go.
    let mut unwind = bcx;
    let mut target;
    loop {
        match unwind.kind {
          block_scope({loop_break: Some(brk), _}) => {
            target = if to_end {
                brk
            } else {
                unwind
            };
            break;
          }
          _ => ()
        }
        unwind = match unwind.parent {
          Some(bcx) => bcx,
          // This is a return from a loop body block
          None => {
            Store(bcx, C_bool(!to_end), bcx.fcx.llretptr);
            cleanup_and_leave(bcx, None, Some(bcx.fcx.llreturn));
            Unreachable(bcx);
            return bcx;
          }
        };
    }
    cleanup_and_Br(bcx, unwind, target.llbb);
    Unreachable(bcx);
    return bcx;
}

fn trans_break(bcx: block) -> block {
    return trans_break_cont(bcx, true);
}

fn trans_cont(bcx: block) -> block {
    return trans_break_cont(bcx, false);
}

fn trans_ret(bcx: block, e: Option<@ast::expr>) -> block {
    let _icx = bcx.insn_ctxt("trans_ret");
    let mut bcx = bcx;
    let retptr = match copy bcx.fcx.loop_ret {
      Some({flagptr, retptr}) => {
        // This is a loop body return. Must set continue flag (our retptr)
        // to false, return flag to true, and then store the value in the
        // parent's retptr.
        Store(bcx, C_bool(true), flagptr);
        Store(bcx, C_bool(false), bcx.fcx.llretptr);
        match e {
          Some(x) => PointerCast(bcx, retptr,
                                 T_ptr(type_of(bcx.ccx(), expr_ty(bcx, x)))),
          None => retptr
        }
      }
      None => bcx.fcx.llretptr
    };
    match e {
      Some(x) => {
        bcx = expr::trans_into(bcx, x, expr::SaveIn(retptr));
      }
      _ => ()
    }
    cleanup_and_leave(bcx, None, Some(bcx.fcx.llreturn));
    Unreachable(bcx);
    return bcx;
}
fn trans_check_expr(bcx: block, chk_expr: @ast::expr,
                    pred_expr: @ast::expr, s: ~str) -> block {
    let _icx = bcx.insn_ctxt("trans_check_expr");
    let expr_str = s + ~" " + expr_to_str(pred_expr, bcx.ccx().sess.intr())
        + ~" failed";
    let Result {bcx, val} = {
        do with_scope_result(bcx, chk_expr.info(), ~"check") |bcx| {
            expr::trans_to_datum(bcx, pred_expr).to_result()
        }
    };
    do with_cond(bcx, Not(bcx, val)) |bcx| {
        trans_fail(bcx, Some(pred_expr.span), expr_str)
    }
}

fn trans_fail_expr(bcx: block,
                   sp_opt: Option<span>,
                   fail_expr: Option<@ast::expr>) -> block {
    let _icx = bcx.insn_ctxt("trans_fail_expr");
    let mut bcx = bcx;
    match fail_expr {
        Some(arg_expr) => {
            let ccx = bcx.ccx(), tcx = ccx.tcx;
            let arg_datum = unpack_datum!(
                bcx, expr::trans_to_datum(bcx, arg_expr));

            if ty::type_is_str(arg_datum.ty) {
                let (lldata, _lllen) = arg_datum.get_base_and_len(bcx);
                return trans_fail_value(bcx, sp_opt, lldata);
            } else if bcx.unreachable || ty::type_is_bot(arg_datum.ty) {
                return bcx;
            } else {
                bcx.sess().span_bug(
                    arg_expr.span, ~"fail called with unsupported type " +
                    ppaux::ty_to_str(tcx, arg_datum.ty));
            }
        }
        _ => return trans_fail(bcx, sp_opt, ~"explicit failure")
    }
}

fn trans_fail(bcx: block, sp_opt: Option<span>, fail_str: ~str)
    -> block
{
    let _icx = bcx.insn_ctxt("trans_fail");
    let V_fail_str = C_cstr(bcx.ccx(), fail_str);
    return trans_fail_value(bcx, sp_opt, V_fail_str);
}

fn trans_fail_value(bcx: block, sp_opt: Option<span>, V_fail_str: ValueRef)
    -> block
{
    let _icx = bcx.insn_ctxt("trans_fail_value");
    let ccx = bcx.ccx();
    let {V_filename, V_line} = match sp_opt {
      Some(sp) => {
        let sess = bcx.sess();
        let loc = codemap::lookup_char_pos(sess.parse_sess.cm, sp.lo);
        {V_filename: C_cstr(bcx.ccx(), loc.file.name),
         V_line: loc.line as int}
      }
      None => {
        {V_filename: C_cstr(bcx.ccx(), ~"<runtime>"),
         V_line: 0}
      }
    };
    let V_str = PointerCast(bcx, V_fail_str, T_ptr(T_i8()));
    let V_filename = PointerCast(bcx, V_filename, T_ptr(T_i8()));
    let args = ~[V_str, V_filename, C_int(ccx, V_line)];
    let bcx = callee::trans_rtcall(bcx, ~"fail_", args, expr::Ignore);
    Unreachable(bcx);
    return bcx;
}
