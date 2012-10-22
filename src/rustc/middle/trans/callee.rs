//!
//
// Handles translation of callees as well as other call-related
// things.  Callees are a superset of normal rust values and sometimes
// have different representations.  In particular, top-level fn items
// and methods are represented as just a fn ptr and not a full
// closure.

use lib::llvm::ValueRef;
use syntax::ast;
use datum::Datum;
use common::{block, node_id_type_params};
use build::*;
use base::{get_item_val, trans_external_path};
use syntax::visit;
use syntax::print::pprust::{expr_to_str, stmt_to_str, path_to_str};
use datum::*;
use util::common::indenter;

// Represents a (possibly monomorphized) top-level fn item or method
// item.  Note that this is just the fn-ptr and is not a Rust closure
// value (which is a pair).
struct FnData {
    llfn: ValueRef,
}

struct MethodData {
    llfn: ValueRef,
    llself: ValueRef,
    self_ty: ty::t,
    self_mode: ast::rmode
}

enum CalleeData {
    Closure(Datum),
    Fn(FnData),
    Method(MethodData)
}

struct Callee {
    bcx: block,
    data: CalleeData
}

fn trans(bcx: block, expr: @ast::expr) -> Callee {
    let _icx = bcx.insn_ctxt("trans_callee");

    // pick out special kinds of expressions that can be called:
    match expr.node {
        ast::expr_path(_) => {
            return trans_def(bcx, bcx.def(expr.id), expr);
        }
        ast::expr_field(base, _, _) => {
            match bcx.ccx().maps.method_map.find(expr.id) {
                Some(origin) => { // An impl method
                    return meth::trans_method_callee(bcx, expr.id,
                                                     base, origin);
                }
                None => {} // not a method, just a field
            }
        }
        _ => {}
    }

    // any other expressions are closures:
    return closure_callee(&expr::trans_to_datum(bcx, expr));

    fn closure_callee(db: &DatumBlock) -> Callee {
        return Callee {bcx: db.bcx, data: Closure(db.datum)};
    }

    fn fn_callee(bcx: block, fd: FnData) -> Callee {
        return Callee {bcx: bcx, data: Fn(fd)};
    }

    fn trans_def(bcx: block, def: ast::def, ref_expr: @ast::expr) -> Callee {
        match def {
            ast::def_fn(did, _) | ast::def_static_method(did, None, _) => {
                fn_callee(bcx, trans_fn_ref(bcx, did, ref_expr.id))
            }
            ast::def_static_method(impl_did, Some(trait_did), _) => {
                fn_callee(bcx, meth::trans_static_method_callee(bcx, impl_did,
                                                                trait_did,
                                                                ref_expr.id))
            }
            ast::def_variant(tid, vid) => {
                // nullary variants are not callable
                assert ty::enum_variant_with_id(bcx.tcx(),
                                                tid,
                                                vid).args.len() > 0u;
                fn_callee(bcx, trans_fn_ref(bcx, vid, ref_expr.id))
            }
            ast::def_arg(*) |
            ast::def_local(*) |
            ast::def_binding(*) |
            ast::def_upvar(*) |
            ast::def_self(*) => {
                closure_callee(&expr::trans_to_datum(bcx, ref_expr))
            }
            ast::def_mod(*) | ast::def_foreign_mod(*) |
            ast::def_const(*) | ast::def_ty(*) | ast::def_prim_ty(*) |
            ast::def_use(*) | ast::def_class(*) | ast::def_typaram_binder(*) |
            ast::def_region(*) | ast::def_label(*) | ast::def_ty_param(*) => {
                bcx.tcx().sess.span_bug(
                    ref_expr.span,
                    fmt!("Cannot translate def %? \
                          to a callable thing!", def));
            }
        }
    }
}

fn trans_fn_ref_to_callee(bcx: block,
                          def_id: ast::def_id,
                          ref_id: ast::node_id) -> Callee
{
    Callee {bcx: bcx,
            data: Fn(trans_fn_ref(bcx, def_id, ref_id))}
}

fn trans_fn_ref(bcx: block,
                def_id: ast::def_id,
                ref_id: ast::node_id) -> FnData {
    /*!
     *
     * Translates a reference (with id `ref_id`) to the fn/method
     * with id `def_id` into a function pointer.  This may require
     * monomorphization or inlining. */

    let _icx = bcx.insn_ctxt("trans_fn");

    let type_params = node_id_type_params(bcx, ref_id);

    let vtables = node_vtables(bcx, ref_id);
    trans_fn_ref_with_vtables(bcx, def_id, ref_id, type_params, vtables)
}

fn trans_fn_ref_with_vtables_to_callee(bcx: block,
                                       def_id: ast::def_id,
                                       ref_id: ast::node_id,
                                       type_params: ~[ty::t],
                                       vtables: Option<typeck::vtable_res>)
    -> Callee
{
    Callee {bcx: bcx,
            data: Fn(trans_fn_ref_with_vtables(bcx, def_id, ref_id,
                                               type_params, vtables))}
}

fn trans_fn_ref_with_vtables(
    bcx: block,            //
    def_id: ast::def_id,   // def id of fn
    ref_id: ast::node_id,  // node id of use of fn
    type_params: ~[ty::t], // values for fn's ty params
    vtables: Option<typeck::vtable_res>)
    -> FnData
{
    //!
    //
    // Translates a reference to a fn/method item, monomorphizing and
    // inlining as it goes.
    //
    // # Parameters
    //
    // - `bcx`: the current block where the reference to the fn occurs
    // - `def_id`: def id of the fn or method item being referenced
    // - `ref_id`: node id of the reference to the fn/method, if applicable.
    //   This parameter may be zero; but, if so, the resulting value may not
    //   have the right type, so it must be cast before being used.
    // - `type_params`: values for each of the fn/method's type parameters
    // - `vtables`: values for each bound on each of the type parameters

    let _icx = bcx.insn_ctxt("trans_fn_with_vtables");
    let ccx = bcx.ccx();
    let tcx = ccx.tcx;

    debug!("trans_fn_ref_with_vtables(bcx=%s, def_id=%?, ref_id=%?, \
            type_params=%?, vtables=%?)",
           bcx.to_str(), def_id, ref_id,
           type_params.map(|t| bcx.ty_to_str(*t)),
           vtables);
    let _indenter = indenter();

    // Polytype of the function item (may have type params)
    let fn_tpt = ty::lookup_item_type(tcx, def_id);

    // Modify the def_id if this is a default method; we want to be
    // monomorphizing the trait's code.
    let (def_id, opt_impl_did) =
            match tcx.provided_method_sources.find(def_id) {
        None => (def_id, None),
        Some(source) => (source.method_id, Some(source.impl_id))
    };

    // Check whether this fn has an inlined copy and, if so, redirect
    // def_id to the local id of the inlined copy.
    let def_id = {
        if def_id.crate != ast::local_crate {
            let may_translate = opt_impl_did.is_none();
            inline::maybe_instantiate_inline(ccx, def_id, may_translate)
        } else {
            def_id
        }
    };

    // We must monomorphise if the fn has type parameters, is a rust
    // intrinsic, or is a default method.  In particular, if we see an
    // intrinsic that is inlined from a different crate, we want to reemit the
    // intrinsic instead of trying to call it in the other crate.
    let must_monomorphise = type_params.len() > 0 ||
        opt_impl_did.is_some() || {
        if def_id.crate == ast::local_crate {
            let map_node = session::expect(
                ccx.sess,
                ccx.tcx.items.find(def_id.node),
                || fmt!("local item should be in ast map"));

            match map_node {
              ast_map::node_foreign_item(
                  _, ast::foreign_abi_rust_intrinsic, _) => true,
              _ => false
            }
        } else {
            false
        }
    };

    // Create a monomorphic verison of generic functions
    if must_monomorphise {
        // Should be either intra-crate or inlined.
        assert def_id.crate == ast::local_crate;

        let mut {val, must_cast} =
            monomorphize::monomorphic_fn(ccx, def_id, type_params,
                                         vtables, opt_impl_did, Some(ref_id));
        if must_cast && ref_id != 0 {
            // Monotype of the REFERENCE to the function (type params
            // are subst'd)
            let ref_ty = common::node_id_type(bcx, ref_id);

            val = PointerCast(
                bcx, val, T_ptr(type_of::type_of_fn_from_ty(ccx, ref_ty)));
        }
        return FnData {llfn: val};
    }

    // Find the actual function pointer.
    let mut val = {
        if def_id.crate == ast::local_crate {
            // Internal reference.
            get_item_val(ccx, def_id.node)
        } else {
            // External reference.
            trans_external_path(ccx, def_id, fn_tpt.ty)
        }
    };

    //NDM I think this is dead. Commenting out to be sure!
    //NDM
    //NDM if tys.len() > 0u {
    //NDM     val = PointerCast(bcx, val, T_ptr(type_of_fn_from_ty(
    //NDM         ccx, node_id_type(bcx, id))));
    //NDM }

    return FnData {llfn: val};
}

// ______________________________________________________________________
// Translating calls

fn trans_call(in_cx: block,
              call_ex: @ast::expr,
              f: @ast::expr,
              args: CallArgs,
              id: ast::node_id,
              dest: expr::Dest)
    -> block
{
    let _icx = in_cx.insn_ctxt("trans_call");
    trans_call_inner(
        in_cx, call_ex.info(), expr_ty(in_cx, f), node_id_type(in_cx, id),
        |cx| trans(cx, f), args, dest, DontAutorefArg)
}

fn trans_rtcall(bcx: block, name: ~str, args: ~[ValueRef], dest: expr::Dest)
    -> block
{
    let did = bcx.ccx().rtcalls[name];
    return trans_rtcall_or_lang_call(bcx, did, args, dest);
}

fn trans_rtcall_or_lang_call(bcx: block, did: ast::def_id, args: ~[ValueRef],
                             dest: expr::Dest) -> block {
    let fty = if did.crate == ast::local_crate {
        ty::node_id_to_type(bcx.ccx().tcx, did.node)
    } else {
        csearch::get_type(bcx.ccx().tcx, did).ty
    };
    let rty = ty::ty_fn_ret(fty);
    return callee::trans_call_inner(
        bcx, None, fty, rty,
        |bcx| trans_fn_ref_with_vtables_to_callee(bcx, did, 0, ~[], None),
        ArgVals(args), dest, DontAutorefArg);
}

fn trans_rtcall_or_lang_call_with_type_params(bcx: block,
                                              did: ast::def_id,
                                              args: ~[ValueRef],
                                              type_params: ~[ty::t],
                                              dest: expr::Dest) -> block {
    let fty;
    if did.crate == ast::local_crate {
        fty = ty::node_id_to_type(bcx.tcx(), did.node);
    } else {
        fty = csearch::get_type(bcx.tcx(), did).ty;
    }

    let rty = ty::ty_fn_ret(fty);
    return callee::trans_call_inner(
        bcx, None, fty, rty,
        |bcx| {
            let callee =
                trans_fn_ref_with_vtables_to_callee(bcx, did, 0, type_params,
                                                    None);

            let new_llval;
            match callee.data {
                Fn(fn_data) => {
                    let substituted = ty::subst_tps(callee.bcx.tcx(),
                                                    type_params,
                                                    None,
                                                    fty);
                    let mut llfnty = type_of::type_of(callee.bcx.ccx(),
                                                      substituted);
                    llfnty = T_ptr(struct_elt(llfnty, 0));
                    new_llval = PointerCast(callee.bcx, fn_data.llfn, llfnty);
                }
                _ => fail
            }
            Callee { bcx: callee.bcx, data: Fn(FnData { llfn: new_llval }) }
        },
        ArgVals(args), dest, DontAutorefArg);
}

fn body_contains_ret(body: ast::blk) -> bool {
    let cx = {mut found: false};
    visit::visit_block(body, cx, visit::mk_vt(@{
        visit_item: |_i, _cx, _v| { },
        visit_expr: |e: @ast::expr, cx: {mut found: bool}, v| {
            if !cx.found {
                match e.node {
                  ast::expr_ret(_) => cx.found = true,
                  _ => visit::visit_expr(e, cx, v),
                }
            }
        },
        ..*visit::default_visitor()
    }));
    cx.found
}

// See [Note-arg-mode]
fn trans_call_inner(
    ++in_cx: block,
    call_info: Option<node_info>,
    fn_expr_ty: ty::t,
    ret_ty: ty::t,
    get_callee: fn(block) -> Callee,
    args: CallArgs,
    dest: expr::Dest,
    autoref_arg: AutorefArg) -> block
{
    do base::with_scope(in_cx, call_info, ~"call") |cx| {
        let ret_in_loop = match args {
          ArgExprs(args) => {
            args.len() > 0u && match vec::last(args).node {
              ast::expr_loop_body(@{
                node: ast::expr_fn_block(_, body, _),
                _
              }) =>  body_contains_ret(body),
              _ => false
            }
          }
          _ => false
        };

        let callee = get_callee(cx);
        let mut bcx = callee.bcx;
        let ccx = cx.ccx();
        let ret_flag = if ret_in_loop {
            let flag = alloca(bcx, T_bool());
            Store(bcx, C_bool(false), flag);
            Some(flag)
        } else { None };

        let (llfn, llenv) = match callee.data {
            Fn(d) => {
                (d.llfn, llvm::LLVMGetUndef(T_opaque_box_ptr(ccx)))
            }
            Method(d) => {
                // Weird but true: we pass self in the *environment* slot!
                let llself = PointerCast(bcx, d.llself,
                                         T_opaque_box_ptr(ccx));
                (d.llfn, llself)
            }
            Closure(d) => {
                // Closures are represented as (llfn, llclosure) pair:
                // load the requisite values out.
                let pair = d.to_ref_llval(bcx);
                let llfn = GEPi(bcx, pair, [0u, abi::fn_field_code]);
                let llfn = Load(bcx, llfn);
                let llenv = GEPi(bcx, pair, [0u, abi::fn_field_box]);
                let llenv = Load(bcx, llenv);
                (llfn, llenv)
            }
        };

        let args_res = trans_args(bcx, llenv, args, fn_expr_ty,
                                  dest, ret_flag, autoref_arg);
        bcx = args_res.bcx;
        let mut llargs = args_res.args;

        let llretslot = args_res.retslot;

        // Now that the arguments have finished evaluating, we need to revoke
        // the cleanup for the self argument, if it exists
        match callee.data {
            Method(d) if d.self_mode == ast::by_copy => {
                revoke_clean(bcx, d.llself);
            }
            _ => {}
        }

        // If the block is terminated, then one or more of the args
        // has type _|_. Since that means it diverges, the code for
        // the call itself is unreachable.
        bcx = base::invoke(bcx, llfn, llargs);
        match dest { // drop the value if it is not being saved.
            expr::Ignore => {
                if llvm::LLVMIsUndef(llretslot) != lib::llvm::True {
                    bcx = glue::drop_ty(bcx, llretslot, ret_ty);
                }
            }
            expr::SaveIn(_) => { }
        }
        if ty::type_is_bot(ret_ty) {
            Unreachable(bcx);
        } else if ret_in_loop {
            bcx = do with_cond(bcx, Load(bcx, ret_flag.get())) |bcx| {
                do option::iter(&copy bcx.fcx.loop_ret) |lret| {
                    Store(bcx, C_bool(true), lret.flagptr);
                    Store(bcx, C_bool(false), bcx.fcx.llretptr);
                }
                base::cleanup_and_leave(bcx, None, Some(bcx.fcx.llreturn));
                Unreachable(bcx);
                bcx
            }
        }
        bcx
    }
}


enum CallArgs {
    ArgExprs(~[@ast::expr]),
    ArgVals(~[ValueRef])
}

fn trans_args(cx: block, llenv: ValueRef, args: CallArgs, fn_ty: ty::t,
              dest: expr::Dest, ret_flag: Option<ValueRef>,
              +autoref_arg: AutorefArg)
    -> {bcx: block, args: ~[ValueRef], retslot: ValueRef}
{
    let _icx = cx.insn_ctxt("trans_args");
    let mut temp_cleanups = ~[];
    let arg_tys = ty::ty_fn_args(fn_ty);
    let mut llargs: ~[ValueRef] = ~[];

    let mut bcx = cx;

    let retty = ty::ty_fn_ret(fn_ty);

    // Arg 0: Output pointer.
    let llretslot = match dest {
        expr::SaveIn(dst) => dst,
        expr::Ignore => {
            if ty::type_is_nil(retty) {
                llvm::LLVMGetUndef(T_ptr(T_nil()))
            } else {
                alloc_ty(bcx, retty)
            }
        }
    };
    llargs.push(llretslot);

    // Arg 1: Env (closure-bindings / self value)
    llargs.push(llenv);

    // ... then explicit args.

    // First we figure out the caller's view of the types of the arguments.
    // This will be needed if this is a generic call, because the callee has
    // to cast her view of the arguments to the caller's view.
    match args {
      ArgExprs(arg_exprs) => {
        let last = arg_exprs.len() - 1u;
        for vec::eachi(arg_exprs) |i, arg_expr| {
            let arg_val = unpack_result!(bcx, {
                trans_arg_expr(bcx, arg_tys[i], *arg_expr, &mut temp_cleanups,
                               if i == last { ret_flag } else { None },
                               autoref_arg)
            });
            llargs.push(arg_val);
        }
      }
      ArgVals(vs) => {
        llargs.push_all(vs);
      }
    }

    // now that all arguments have been successfully built, we can revoke any
    // temporary cleanups, as they are only needed if argument construction
    // should fail (for example, cleanup of copy mode args).
    for vec::each(temp_cleanups) |c| {
        revoke_clean(bcx, *c)
    }

    return {bcx: bcx, args: llargs, retslot: llretslot};
}

enum AutorefArg {
    DontAutorefArg,
    DoAutorefArg
}

// temp_cleanups: cleanups that should run only if failure occurs before the
// call takes place:
fn trans_arg_expr(bcx: block,
                  formal_ty: ty::arg,
                  arg_expr: @ast::expr,
                  temp_cleanups: &mut ~[ValueRef],
                  ret_flag: Option<ValueRef>,
                  +autoref_arg: AutorefArg)
    -> Result
{
    let _icx = bcx.insn_ctxt("trans_arg_expr");
    let ccx = bcx.ccx();

    debug!("trans_arg_expr(formal_ty=(%?,%s), arg_expr=%s, \
            ret_flag=%?)",
           formal_ty.mode, bcx.ty_to_str(formal_ty.ty),
           bcx.expr_to_str(arg_expr),
           ret_flag.map(|v| bcx.val_str(*v)));
    let _indenter = indenter();

    // translate the arg expr to a datum
    let arg_datumblock = match ret_flag {
        None => expr::trans_to_datum(bcx, arg_expr),

        // If there is a ret_flag, this *must* be a loop body
        Some(_) => {
            match arg_expr.node {
                ast::expr_loop_body(
                    blk @ @{node:ast::expr_fn_block(decl, body, cap), _}) =>
                {
                    let scratch_ty = expr_ty(bcx, blk);
                    let scratch = alloc_ty(bcx, scratch_ty);
                    let arg_ty = expr_ty(bcx, arg_expr);
                    let proto = ty::ty_fn_proto(arg_ty);
                    let bcx = closure::trans_expr_fn(
                        bcx, proto, decl, body, blk.id,
                        cap, Some(ret_flag), expr::SaveIn(scratch));
                    DatumBlock {bcx: bcx,
                                datum: Datum {val: scratch,
                                              ty: scratch_ty,
                                              mode: ByRef,
                                              source: FromRvalue}}
                }
                _ => {
                    bcx.sess().impossible_case(
                        arg_expr.span, ~"ret_flag with non-loop-\
                                         body expr");
                }
            }
        }
    };
    let mut arg_datum = arg_datumblock.datum;
    let mut bcx = arg_datumblock.bcx;

    debug!("   arg datum: %s", arg_datum.to_str(bcx.ccx()));

    // finally, deal with the various modes
    let arg_mode = ty::resolved_mode(ccx.tcx, formal_ty.mode);
    let mut val;
    if ty::type_is_bot(arg_datum.ty) {
        // For values of type _|_, we generate an
        // "undef" value, as such a value should never
        // be inspected. It's important for the value
        // to have type lldestty (the callee's expected type).
        let llformal_ty = type_of::type_of(ccx, formal_ty.ty);
        val = llvm::LLVMGetUndef(llformal_ty);
    } else {
        // FIXME(#3548) use the adjustments table
        match autoref_arg {
            DoAutorefArg => { val = arg_datum.to_ref_llval(bcx); }
            DontAutorefArg => {
                match arg_mode {
                    ast::by_ref => {
                        val = arg_datum.to_ref_llval(bcx);
                    }

                    ast::by_val => {
                        // NB: avoid running the take glue.
                        val = arg_datum.to_value_llval(bcx);
                    }

                    ast::by_copy | ast::by_move => {
                        let scratch = scratch_datum(bcx, arg_datum.ty, false);

                        if arg_mode == ast::by_move {
                            // NDM---Doesn't seem like this should be
                            // necessary
                            if !arg_datum.store_will_move() {
                                bcx.sess().span_bug(
                                    arg_expr.span,
                                    fmt!("move mode but datum will not \
                                          store: %s",
                                          arg_datum.to_str(bcx.ccx())));
                            }
                        }

                        arg_datum.store_to_datum(bcx, INIT, scratch);

                        // Technically, ownership of val passes to the callee.
                        // However, we must cleanup should we fail before the
                        // callee is actually invoked.
                        scratch.add_clean(bcx);
                        temp_cleanups.push(scratch.val);

                        match arg_datum.appropriate_mode() {
                            ByValue => {
                                val = Load(bcx, scratch.val);
                            }
                            ByRef => {
                                val = scratch.val;
                            }
                        }
                    }
                }
            }
        }

        if formal_ty.ty != arg_datum.ty {
            // this could happen due to e.g. subtyping
            let llformal_ty = type_of::type_of_explicit_arg(ccx, formal_ty);
            debug!("casting actual type (%s) to match formal (%s)",
                   bcx.val_str(val), bcx.llty_str(llformal_ty));
            val = PointerCast(bcx, val, llformal_ty);
        }
    }

    debug!("--- trans_arg_expr passing %s", val_str(bcx.ccx().tn, val));
    return rslt(bcx, val);
}

