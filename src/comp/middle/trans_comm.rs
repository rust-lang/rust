/**
   Translation for various task and comm-related things.

   Most of this will probably go away as we move more of this into
   libraries.

*/

import std::str;
import std::option;
import option::none;
import option::some;

import lib::llvm::llvm;
import lib::llvm::llvm::ValueRef;

import util::ppaux::ty_to_str;
import syntax::print::pprust::expr_to_str;
import syntax::ast;
import back::link::mangle_internal_name_by_path_and_seq;

import trans_common::*;
import trans::*;

export trans_port;
export trans_chan;
export trans_spawn;
export trans_send;
export trans_recv;

fn trans_port(cx: &@block_ctxt, id: ast::node_id) -> result {
    let t = node_id_type(cx.fcx.lcx.ccx, id);
    let unit_ty;
    alt ty::struct(cx.fcx.lcx.ccx.tcx, t) {
      ty::ty_port(t) { unit_ty = t; }
      _ { cx.fcx.lcx.ccx.sess.bug("non-port type in trans_port"); }
    }
    let bcx = cx;
    let unit_sz = size_of(bcx, unit_ty);
    bcx = unit_sz.bcx;
    let port_raw_val =
        bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.new_port,
                       ~[bcx.fcx.lltaskptr, unit_sz.val]);
    let llty = type_of(cx.fcx.lcx.ccx, cx.sp, t);
    let port_val = bcx.build.PointerCast(port_raw_val, llty);
    add_clean_temp(bcx, port_val, t);
    ret rslt(bcx, port_val);
}

fn trans_chan(cx: &@block_ctxt, e: &@ast::expr, id: ast::node_id) -> result {
    let bcx = cx;
    let prt = trans_expr(bcx, e);
    bcx = prt.bcx;
    let prt_val = bcx.build.PointerCast(prt.val, T_opaque_port_ptr());
    let chan_raw_val =
        bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.new_chan,
                       ~[bcx.fcx.lltaskptr, prt_val]);
    let chan_ty = node_id_type(bcx.fcx.lcx.ccx, id);
    let chan_llty = type_of(bcx.fcx.lcx.ccx, e.span, chan_ty);
    let chan_val = bcx.build.PointerCast(chan_raw_val, chan_llty);
    add_clean_temp(bcx, chan_val, chan_ty);
    ret rslt(bcx, chan_val);
}

fn trans_spawn(cx: &@block_ctxt, dom: &ast::spawn_dom, name: &option::t[str],
               func: &@ast::expr, args: &[@ast::expr], id: ast::node_id) ->
   result {
    let bcx = cx;
    // Make the task name

    let tname =
        alt name {
          none. {
            let argss = std::ivec::map(expr_to_str, args);
            #fmt("%s(%s)", expr_to_str(func), str::connect(argss, ", "))
          }
          some(n) { n }
        };
    // Generate code
    //
    // This is a several step process. The following things need to happen
    // (not necessarily in order):
    //
    // 1. Evaluate all the arguments to the spawnee.
    //
    // 2. Alloca a tuple that holds these arguments (they must be in reverse
    // order, so that they match the expected stack layout for the spawnee)
    //
    // 3. Fill the tuple with the arguments we evaluated.
    //
    // 3.5. Generate a wrapper function that takes the tuple and unpacks it to
    // call the real task.
    //
    // 4. Pass a pointer to the wrapper function and the argument tuple to
    // upcall_start_task. In order to do this, we need to allocate another
    // tuple that matches the arguments expected by rust_task::start.
    //
    // 5. Oh yeah, we have to create the task before we start it...

    // But first, we'll create a task.

    let lltname: ValueRef = C_str(bcx.fcx.lcx.ccx, tname);
    let new_task =
        bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.new_task,
                       ~[bcx.fcx.lltaskptr, lltname]);

    // Translate the arguments, remembering their types and where the values
    // ended up.

    let arg_tys: [ty::t] = ~[];
    let arg_vals: [ValueRef] = ~[];
    for e: @ast::expr  in args {
        let e_ty = ty::expr_ty(cx.fcx.lcx.ccx.tcx, e);
        let arg = trans_expr(bcx, e);

        arg = deep_copy(arg.bcx, arg.val, e_ty, new_task);

        bcx = arg.bcx;

        arg_vals += ~[arg.val];
        arg_tys += ~[e_ty];
    }
    // Make the tuple.

    let args_ty = ty::mk_imm_tup(cx.fcx.lcx.ccx.tcx, arg_tys);
    // Allocate and fill the tuple.

    let llargs = alloc_ty(bcx, args_ty);
    let i = 0u;
    for v: ValueRef  in arg_vals {
        let target = bcx.build.GEP(llargs.val, ~[C_int(0), C_int(i as int)]);

        bcx.build.Store(v, target);
        i += 1u;
    }

    // Generate the wrapper function
    let wrapper = mk_spawn_wrapper(bcx, func, args_ty);
    bcx = wrapper.bcx;
    let llfnptr_i = bcx.build.PointerCast(wrapper.val, T_int());

    // And start the task
    let llargs_i = bcx.build.PointerCast(llargs.val, T_int());
    let args_size = size_of(bcx, args_ty).val;
    bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.start_task,
                   ~[bcx.fcx.lltaskptr, new_task, llfnptr_i, llargs_i,
                     args_size]);
    let task_ty = node_id_type(bcx.fcx.lcx.ccx, id);
    add_clean_temp(bcx, new_task, task_ty);
    ret rslt(bcx, new_task);
}

fn mk_spawn_wrapper(cx: &@block_ctxt, func: &@ast::expr, args_ty: &ty::t) ->
   result {
    let llmod = cx.fcx.lcx.ccx.llmod;
    let wrapper_fn_type =
        type_of_fn(cx.fcx.lcx.ccx, cx.sp, ast::proto_fn,
                   ~[{mode: ty::mo_alias(false), ty: args_ty}], ty::idx_nil,
                   0u);
    // TODO: construct a name based on tname

    let wrap_name: str =
        mangle_internal_name_by_path_and_seq(cx.fcx.lcx.ccx, cx.fcx.lcx.path,
                                             "spawn_wrapper");
    let llfndecl = decl_cdecl_fn(llmod, wrap_name, wrapper_fn_type);
    let fcx = new_fn_ctxt(cx.fcx.lcx, cx.sp, llfndecl);
    let fbcx = new_top_block_ctxt(fcx);
    // 3u to skip the three implicit args

    let arg: ValueRef = llvm::LLVMGetParam(fcx.llfn, 3u);
    let child_args: [ValueRef] =
        ~[llvm::LLVMGetParam(fcx.llfn, 0u), llvm::LLVMGetParam(fcx.llfn, 1u),
          llvm::LLVMGetParam(fcx.llfn, 2u)];
    // unpack the arguments

    alt ty::struct(fcx.lcx.ccx.tcx, args_ty) {
      ty::ty_rec(fields) {
        let i = 0;
        for f: ty::field  in fields {
            let src = fbcx.build.GEP(arg, ~[C_int(0), C_int(i)]);
            i += 1;
            let child_arg = fbcx.build.Load(src);
            child_args += ~[child_arg];
        }
      }
    }
    // Find the function

    let fnptr = trans_lval(fbcx, func).res;
    fbcx = fnptr.bcx;
    let llfnptr = fbcx.build.GEP(fnptr.val, ~[C_int(0), C_int(0)]);
    let llfn = fbcx.build.Load(llfnptr);
    fbcx.build.FastCall(llfn, child_args);
    fbcx.build.RetVoid();
    finish_fn(fcx, fbcx.llbb);
    // TODO: make sure we clean up everything we need to.

    ret rslt(cx, llfndecl);
}

fn trans_send(cx: &@block_ctxt, lhs: &@ast::expr, rhs: &@ast::expr,
              id: ast::node_id) -> result {
    let bcx = cx;
    let chn = trans_expr(bcx, lhs);
    bcx = chn.bcx;
    let data = trans_lval(bcx, rhs);
    bcx = data.res.bcx;
    let chan_ty = node_id_type(cx.fcx.lcx.ccx, id);
    let unit_ty;
    alt ty::struct(cx.fcx.lcx.ccx.tcx, chan_ty) {
      ty::ty_chan(t) { unit_ty = t; }
      _ { bcx.fcx.lcx.ccx.sess.bug("non-chan type in trans_send"); }
    }
    let data_alloc = alloc_ty(bcx, unit_ty);
    bcx = data_alloc.bcx;
    let data_tmp = move_val_if_temp(bcx, INIT, data_alloc.val, data, unit_ty);
    bcx = data_tmp.bcx;
    let llchanval = bcx.build.PointerCast(chn.val, T_opaque_chan_ptr());
    let lldataptr = bcx.build.PointerCast(data_alloc.val, T_ptr(T_i8()));
    bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.send,
                   ~[bcx.fcx.lltaskptr, llchanval, lldataptr]);

    // Deinit the stuff we sent.
    bcx = zero_alloca(bcx, data_alloc.val, unit_ty).bcx;

    ret rslt(bcx, chn.val);
}

fn trans_recv(cx: &@block_ctxt, lhs: &@ast::expr, rhs: &@ast::expr,
              id: ast::node_id) -> result {
    let bcx = cx;
    // FIXME: calculate copy init-ness in typestate.

    let unit_ty = node_id_type(cx.fcx.lcx.ccx, id);
    let tmp_alloc = alloc_ty(bcx, unit_ty);
    bcx = tmp_alloc.bcx;

    let prt = trans_expr(bcx, lhs);
    bcx = prt.bcx;
    let lldataptr = bcx.build.PointerCast(tmp_alloc.val,
                                          T_ptr(T_ptr(T_i8())));
    let llportptr = bcx.build.PointerCast(prt.val, T_opaque_port_ptr());
    bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.recv,
                   ~[bcx.fcx.lltaskptr, lldataptr, llportptr]);

    let tmp = load_if_immediate(bcx, tmp_alloc.val, unit_ty);

    let data = trans_lval(bcx, rhs);
    assert (data.is_mem);
    bcx = data.res.bcx;

    let tmp_lval = lval_val(bcx, tmp);

    let recv_res =
        move_val(bcx, DROP_EXISTING, data.res.val, tmp_lval, unit_ty);

    ret rslt(recv_res.bcx, recv_res.val);
}

// Does a deep copy of a value. This is needed for passing arguments to child
// tasks, and for sending things through channels. There are probably some
// uniqueness optimizations and things we can do here for tasks in the same
// domain.
fn deep_copy(bcx: &@block_ctxt, v: ValueRef, t: ty::t, target_task: ValueRef)
   -> result {
    // TODO: make sure all paths add any reference counting that they need to.

    // TODO: Teach deep copy to understand everything else it needs to.

    let tcx = bcx.fcx.lcx.ccx.tcx;
    if ty::type_is_scalar(tcx, t) {
        ret rslt(bcx, v);
    } else if (ty::type_is_str(tcx, t)) {
        ret rslt(bcx,
                 bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.dup_str,
                                ~[bcx.fcx.lltaskptr, target_task, v]));
    } else if (ty::type_is_chan(tcx, t)) {
        // If this is a channel, we need to clone it.
        let chan_ptr = bcx.build.PointerCast(v, T_opaque_chan_ptr());

        let chan_raw_val =
            bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.clone_chan,
                           ~[bcx.fcx.lltaskptr, target_task, chan_ptr]);

        // Cast back to the type the context was expecting.
        let chan_val = bcx.build.PointerCast(chan_raw_val, val_ty(v));

        ret rslt(bcx, chan_val);
    } else if (ty::type_is_structural(tcx, t)) {
        fn inner_deep_copy(bcx: &@block_ctxt, v: ValueRef, t: ty::t) ->
           result {
            log_err "Unimplemented type for deep_copy.";
            fail;
        }

        ret iter_structural_ty(bcx, v, t, inner_deep_copy);
    } else {
        bcx.fcx.lcx.ccx.sess.bug("unexpected type in " + "trans::deep_copy: "
                                     + ty_to_str(tcx, t));
    }
}

