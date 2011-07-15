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

fn trans_port(&@block_ctxt cx, ast::node_id id) -> result {
    auto t = node_id_type(cx.fcx.lcx.ccx, id);
    auto unit_ty;
    alt (ty::struct(cx.fcx.lcx.ccx.tcx, t)) {
        case (ty::ty_port(?t)) { unit_ty = t; }
        case (_) { cx.fcx.lcx.ccx.sess.bug("non-port type in trans_port"); }
    }
    auto bcx = cx;
    auto unit_sz = size_of(bcx, unit_ty);
    bcx = unit_sz.bcx;
    auto port_raw_val =
        bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.new_port,
                       ~[bcx.fcx.lltaskptr, unit_sz.val]);
    auto llty = type_of(cx.fcx.lcx.ccx, cx.sp, t);
    auto port_val = bcx.build.PointerCast(port_raw_val, llty);
    add_clean_temp(bcx, port_val, t);
    ret rslt(bcx, port_val);
}

fn trans_chan(&@block_ctxt cx, &@ast::expr e, ast::node_id id) -> result {
    auto bcx = cx;
    auto prt = trans_expr(bcx, e);
    bcx = prt.bcx;
    auto prt_val = bcx.build.PointerCast(prt.val, T_opaque_port_ptr());
    auto chan_raw_val =
        bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.new_chan,
                       ~[bcx.fcx.lltaskptr, prt_val]);
    auto chan_ty = node_id_type(bcx.fcx.lcx.ccx, id);
    auto chan_llty = type_of(bcx.fcx.lcx.ccx, e.span, chan_ty);
    auto chan_val = bcx.build.PointerCast(chan_raw_val, chan_llty);
    add_clean_temp(bcx, chan_val, chan_ty);
    ret rslt(bcx, chan_val);
}

fn trans_spawn(&@block_ctxt cx, &ast::spawn_dom dom, &option::t[str] name,
               &@ast::expr func, &(@ast::expr)[] args, ast::node_id id)
        -> result {
    auto bcx = cx;
    // Make the task name

    auto tname =
        alt (name) {
            case (none) {
                auto argss = std::ivec::map(expr_to_str, args);
                #fmt("%s(%s)", expr_to_str(func),
                     str::connect_ivec(argss, ", "))
            }
            case (some(?n)) { n }
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

    let ValueRef lltname = C_str(bcx.fcx.lcx.ccx, tname);
    auto new_task =
        bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.new_task,
                       ~[bcx.fcx.lltaskptr, lltname]);

    // Translate the arguments, remembering their types and where the values
    // ended up.

    let ty::t[] arg_tys = ~[];
    let ValueRef[] arg_vals = ~[];
    for (@ast::expr e in args) {
        auto e_ty = ty::expr_ty(cx.fcx.lcx.ccx.tcx, e);
        auto arg = trans_expr(bcx, e);

        arg = deep_copy(arg.bcx, arg.val, e_ty, new_task);

        bcx = arg.bcx;

        arg_vals += ~[arg.val];
        arg_tys += ~[e_ty];
    }
    // Make the tuple.

    auto args_ty = ty::mk_imm_tup(cx.fcx.lcx.ccx.tcx, arg_tys);
    // Allocate and fill the tuple.

    auto llargs = alloc_ty(bcx, args_ty);
    auto i = 0u;
    for (ValueRef v in arg_vals) {
        // log_err #fmt("ty(llargs) = %s",
        //              val_str(bcx.fcx.lcx.ccx.tn, llargs.val));

        auto target = bcx.build.GEP(llargs.val, ~[C_int(0), C_int(i as int)]);
        // log_err #fmt("ty(v) = %s", val_str(bcx.fcx.lcx.ccx.tn, v));
        // log_err #fmt("ty(target) = %s",
        //              val_str(bcx.fcx.lcx.ccx.tn, target));

        bcx.build.Store(v, target);
        i += 1u;
    }

    // Generate the wrapper function
    auto wrapper = mk_spawn_wrapper(bcx, func, args_ty);
    bcx = wrapper.bcx;
    auto llfnptr_i = bcx.build.PointerCast(wrapper.val, T_int());

    // And start the task
    auto llargs_i = bcx.build.PointerCast(llargs.val, T_int());
    auto args_size = size_of(bcx, args_ty).val;
    bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.start_task,
                   ~[bcx.fcx.lltaskptr, new_task, llfnptr_i, llargs_i,
                     args_size]);
    auto task_ty = node_id_type(bcx.fcx.lcx.ccx, id);
    add_clean_temp(bcx, new_task, task_ty);
    ret rslt(bcx, new_task);
}

fn mk_spawn_wrapper(&@block_ctxt cx, &@ast::expr func, &ty::t args_ty) ->
   result {
    auto llmod = cx.fcx.lcx.ccx.llmod;
    auto wrapper_fn_type =
        type_of_fn(cx.fcx.lcx.ccx, cx.sp, ast::proto_fn,
                   ~[rec(mode=ty::mo_alias(false), ty=args_ty)], ty::idx_nil,
                   0u);
    // TODO: construct a name based on tname

    let str wrap_name =
        mangle_internal_name_by_path_and_seq(cx.fcx.lcx.ccx, cx.fcx.lcx.path,
                                             "spawn_wrapper");
    auto llfndecl = decl_cdecl_fn(llmod, wrap_name, wrapper_fn_type);
    auto fcx = new_fn_ctxt(cx.fcx.lcx, cx.sp, llfndecl);
    auto fbcx = new_top_block_ctxt(fcx);
    // 3u to skip the three implicit args

    let ValueRef arg = llvm::LLVMGetParam(fcx.llfn, 3u);
    let ValueRef[] child_args =
        ~[llvm::LLVMGetParam(fcx.llfn, 0u), llvm::LLVMGetParam(fcx.llfn, 1u),
          llvm::LLVMGetParam(fcx.llfn, 2u)];
    // unpack the arguments

    alt (ty::struct(fcx.lcx.ccx.tcx, args_ty)) {
        case (ty::ty_tup(?elements)) {
            auto i = 0;
            for (ty::mt m in elements) {
                auto src = fbcx.build.GEP(arg, ~[C_int(0), C_int(i)]);
                i += 1;
                auto child_arg = fbcx.build.Load(src);
                child_args += ~[child_arg];
            }
        }
    }
    // Find the function

    auto fnptr = trans_lval(fbcx, func).res;
    fbcx = fnptr.bcx;
    auto llfnptr = fbcx.build.GEP(fnptr.val, ~[C_int(0), C_int(0)]);
    auto llfn = fbcx.build.Load(llfnptr);
    fbcx.build.FastCall(llfn, child_args);
    fbcx.build.RetVoid();
    finish_fn(fcx, fbcx.llbb);
    // TODO: make sure we clean up everything we need to.

    ret rslt(cx, llfndecl);
}

// Does a deep copy of a value. This is needed for passing arguments to child
// tasks, and for sending things through channels. There are probably some
// uniqueness optimizations and things we can do here for tasks in the same
// domain.
fn deep_copy(&@block_ctxt bcx, ValueRef v, ty::t t, ValueRef target_task)
    -> result
{
    // TODO: make sure all paths add any reference counting that they need to.

    // TODO: Teach deep copy to understand everything else it needs to.

    auto tcx = bcx.fcx.lcx.ccx.tcx;
    if(ty::type_is_scalar(tcx, t)) {
        ret rslt(bcx, v);
    }
    else if(ty::type_is_str(tcx, t)) {
        ret rslt(bcx,
                bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.dup_str,
                               ~[bcx.fcx.lltaskptr, target_task, v]));
    }
    else if(ty::type_is_chan(tcx, t)) {
        // If this is a channel, we need to clone it.
        auto chan_ptr = bcx.build.PointerCast(v, T_opaque_chan_ptr());

        auto chan_raw_val =
            bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.clone_chan,
                           ~[bcx.fcx.lltaskptr, target_task, chan_ptr]);

        // Cast back to the type the context was expecting.
        auto chan_val = bcx.build.PointerCast(chan_raw_val,
                                              val_ty(v));

        ret rslt(bcx, chan_val);
    }
    else if(ty::type_is_structural(tcx, t)) {
        fn inner_deep_copy(&@block_ctxt bcx, ValueRef v, ty::t t) -> result {
            log_err "Unimplemented type for deep_copy.";
            fail;
        }

        ret iter_structural_ty(bcx, v, t, inner_deep_copy);
    }
    else {
        bcx.fcx.lcx.ccx.sess.bug("unexpected type in " +
                                 "trans::deep_copy: " +
                                 ty_to_str(tcx, t));
    }
}

fn trans_send(&@block_ctxt cx, &@ast::expr lhs, &@ast::expr rhs,
              ast::node_id id) -> result {
    auto bcx = cx;
    auto chn = trans_expr(bcx, lhs);
    bcx = chn.bcx;
    auto data = trans_lval(bcx, rhs);
    bcx = data.res.bcx;
    auto chan_ty = node_id_type(cx.fcx.lcx.ccx, id);
    auto unit_ty;
    alt (ty::struct(cx.fcx.lcx.ccx.tcx, chan_ty)) {
        case (ty::ty_chan(?t)) { unit_ty = t; }
        case (_) { bcx.fcx.lcx.ccx.sess.bug("non-chan type in trans_send"); }
    }
    auto data_alloc = alloc_ty(bcx, unit_ty);
    bcx = data_alloc.bcx;
    auto data_tmp = move_val_if_temp(bcx, INIT, data_alloc.val,
                                     data, unit_ty);
    bcx = data_tmp.bcx;
    add_clean_temp(bcx, data_alloc.val, unit_ty);
    auto llchanval = bcx.build.PointerCast(chn.val, T_opaque_chan_ptr());
    auto lldataptr = bcx.build.PointerCast(data_alloc.val, T_ptr(T_i8()));
    bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.send,
                   ~[bcx.fcx.lltaskptr, llchanval, lldataptr]);
    ret rslt(bcx, chn.val);
}

fn trans_recv(&@block_ctxt cx, &@ast::expr lhs, &@ast::expr rhs,
              ast::node_id id) -> result {
    auto bcx = cx;
    auto data = trans_lval(bcx, rhs);
    assert (data.is_mem);
    bcx = data.res.bcx;
    auto unit_ty = node_id_type(bcx.fcx.lcx.ccx, id);
    // FIXME: calculate copy init-ness in typestate.

    ret recv_val(bcx, data.res.val, lhs, unit_ty, DROP_EXISTING);
}

fn recv_val(&@block_ctxt cx, ValueRef to, &@ast::expr from, &ty::t unit_ty,
            copy_action action) -> result {
    auto bcx = cx;
    auto prt = trans_expr(bcx, from);
    bcx = prt.bcx;
    auto lldataptr = bcx.build.PointerCast(to, T_ptr(T_ptr(T_i8())));
    auto llportptr = bcx.build.PointerCast(prt.val, T_opaque_port_ptr());
    bcx.build.Call(bcx.fcx.lcx.ccx.upcalls.recv,
                   ~[bcx.fcx.lltaskptr, lldataptr, llportptr]);
    auto data_load = load_if_immediate(bcx, to, unit_ty);
    auto cp = copy_val(bcx, action, to, data_load, unit_ty);
    bcx = cp.bcx;
    // TODO: Any cleanup need to be done here?
    ret rslt(bcx, to);
}

