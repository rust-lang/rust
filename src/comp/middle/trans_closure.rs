import syntax::ast;
import syntax::ast_util;
import lib::llvm::llvm::ValueRef;
import trans_common::*;
import trans_build::*;
import trans::*;
import middle::freevars::get_freevars;
import option::{some, none};
import back::abi;
import syntax::codemap::span;
import back::link::mangle_internal_name_by_path;
import trans::{
    trans_shared_malloc,
    type_of_inner,
    size_of,
    node_id_type,
    INIT,
    trans_shared_free,
    drop_ty,
    new_sub_block_ctxt,
    load_if_immediate,
    dest
};

tag environment_value {
    env_expr(@ast::expr);
    env_direct(ValueRef, ty::t, bool);
}

// Given a block context and a list of tydescs and values to bind
// construct a closure out of them. If copying is true, it is a
// heap allocated closure that copies the upvars into environment.
// Otherwise, it is stack allocated and copies pointers to the upvars.
fn build_environment(bcx: @block_ctxt, lltydescs: [ValueRef],
                     bound_values: [environment_value],
                     mode: closure_constr_mode) ->
   {ptr: ValueRef, ptrty: ty::t, bcx: @block_ctxt} {

    fn dummy_environment_box(bcx: @block_ctxt, r: result)
        -> (@block_ctxt, ValueRef, ValueRef) {
        // Prevent glue from trying to free this.
        let ccx = bcx_ccx(bcx);
        let ref_cnt = GEPi(bcx, r.val, [0, abi::box_rc_field_refcnt]);
        Store(r.bcx, C_int(ccx, 2), ref_cnt);
        let closure = GEPi(r.bcx, r.val, [0, abi::box_rc_field_body]);
        (r.bcx, closure, r.val)
    }

    fn clone_tydesc(bcx: @block_ctxt,
                    mode: closure_constr_mode,
                    td: ValueRef) -> ValueRef {
        ret alt mode {
          for_block. | for_closure. { td }
          for_send. { Call(bcx, bcx_ccx(bcx).upcalls.clone_type_desc, [td]) }
        };
    }

    //let ccx = bcx_ccx(bcx);
    let tcx = bcx_tcx(bcx);

    // First, synthesize a tuple type containing the types of all the
    // bound expressions.
    // bindings_ty = [bound_ty1, bound_ty2, ...]
    let bound_tys = [];
    for bv in bound_values {
        bound_tys += [alt bv {
          env_direct(_, t, _) { t }
          env_expr(e) { ty::expr_ty(tcx, e) }
        }];
    }
    let bindings_ty: ty::t = ty::mk_tup(tcx, bound_tys);

    // NB: keep this in sync with T_closure_ptr; we're making
    // a ty::t structure that has the same "shape" as the LLVM type
    // it constructs.

    // Make a vector that contains ty_param_count copies of tydesc_ty.
    // (We'll need room for that many tydescs in the closure.)
    let ty_param_count = vec::len(lltydescs);
    let tydesc_ty: ty::t = ty::mk_type(tcx);
    let captured_tys: [ty::t] = vec::init_elt(tydesc_ty, ty_param_count);

    // Get all the types we've got (some of which we synthesized
    // ourselves) into a vector.  The whole things ends up looking
    // like:

    // closure_ty = (
    //   tydesc_ty, (bound_ty1, bound_ty2, ...),
    //   /*int,*/ (tydesc_ty, tydesc_ty, ...))
    let closure_tys: [ty::t] =
        [tydesc_ty, bindings_ty,
         /*ty::mk_uint(tcx),*/ ty::mk_tup(tcx, captured_tys)];
    let closure_ty: ty::t = ty::mk_tup(tcx, closure_tys);

    let temp_cleanups = [];

    // Allocate a box that can hold something closure-sized.
    //
    // For now, no matter what kind of closure we have, we always allocate
    // space for a ref cnt in the closure.  If the closure is a block or
    // unique closure, this ref count isn't really used: we initialize it to 2
    // so that it will never drop to zero.  This is a hack and could go away
    // but then we'd have to modify the code to do the right thing when
    // casting from a shared closure to a block.
    let (bcx, closure, box) = alt mode {
      for_closure. {
        let r = trans::trans_malloc_boxed(bcx, closure_ty);
        add_clean_free(bcx, r.box, false);
        temp_cleanups += [r.box];
        (r.bcx, r.body, r.box)
      }
      for_send. {
        // Dummy up a box in the exchange heap.
        let tup_ty = ty::mk_tup(tcx, [ty::mk_int(tcx), closure_ty]);
        let box_ty = ty::mk_uniq(tcx, {ty: tup_ty, mut: ast::imm});
        check trans_uniq::type_is_unique_box(bcx, box_ty);
        let r = trans_uniq::alloc_uniq(bcx, box_ty);
        add_clean_free(bcx, r.val, true);
        dummy_environment_box(bcx, r)
      }
      for_block. {
        // Dummy up a box on the stack,
        let ty = ty::mk_tup(tcx, [ty::mk_int(tcx), closure_ty]);
        let r = trans::alloc_ty(bcx, ty);
        dummy_environment_box(bcx, r)
      }
    };

    // Store bindings tydesc.
    alt mode {
      for_closure. | for_send. {
        let bound_tydesc = GEPi(bcx, closure, [0, abi::closure_elt_tydesc]);
        let ti = none;
        let {result:bindings_tydesc, _} =
            trans::get_tydesc(bcx, bindings_ty, true, trans::tps_normal, ti);
        trans::lazily_emit_tydesc_glue(bcx, abi::tydesc_field_drop_glue, ti);
        trans::lazily_emit_tydesc_glue(bcx, abi::tydesc_field_free_glue, ti);
        bcx = bindings_tydesc.bcx;
        let td = clone_tydesc(bcx, mode, bindings_tydesc.val);
        Store(bcx, td, bound_tydesc);
      }
      for_block. {}
    }

    // Copy expr values into boxed bindings.
    // Silly check
    check type_is_tup_like(bcx, closure_ty);
    let closure_box = box;
    let closure_box_ty = ty::mk_imm_box(bcx_tcx(bcx), closure_ty);
    let i = 0u;
    for bv in bound_values {
        let bound = trans::GEP_tup_like_1(bcx, closure_box_ty, closure_box,
                                          [0, abi::box_rc_field_body,
                                           abi::closure_elt_bindings,
                                           i as int]);
        bcx = bound.bcx;
        alt bv {
          env_expr(e) {
            bcx = trans::trans_expr_save_in(bcx, e, bound.val);
            add_clean_temp_mem(bcx, bound.val, bound_tys[i]);
            temp_cleanups += [bound.val];
          }
          env_direct(val, ty, is_mem) {
            alt mode {
              for_closure. | for_send. {
                let val1 = is_mem ? load_if_immediate(bcx, val, ty) : val;
                bcx = trans::copy_val(bcx, INIT, bound.val, val1, ty);
              }
              for_block. {
                let addr = is_mem ? val : do_spill_noroot(bcx, val);
                Store(bcx, addr, bound.val);
              }
            }
          }
        }
        i += 1u;
    }
    for cleanup in temp_cleanups { revoke_clean(bcx, cleanup); }

    // If necessary, copy tydescs describing type parameters into the
    // appropriate slot in the closure.
    // Silly check as well
    //check type_is_tup_like(bcx, closure_ty);
    //let {bcx:bcx, val:n_ty_params_slot} =
    //    GEP_tup_like(bcx, closure_ty, closure,
    //                 [0, abi::closure_elt_n_ty_params]);
    //Store(bcx, C_uint(ccx, vec::len(lltydescs)), n_ty_params_slot);
    check type_is_tup_like(bcx, closure_ty);
    let {bcx:bcx, val:ty_params_slot} =
        GEP_tup_like(bcx, closure_ty, closure,
                     [0, abi::closure_elt_ty_params]);
    i = 0u;
    for td: ValueRef in lltydescs {
        let ty_param_slot = GEPi(bcx, ty_params_slot, [0, i as int]);
        let cloned_td = clone_tydesc(bcx, mode, td);
        Store(bcx, cloned_td, ty_param_slot);
        i += 1u;
    }

    ret {ptr: box, ptrty: closure_ty, bcx: bcx};
}

tag closure_constr_mode {
    for_block;
    for_closure;
    for_send;
}

// Given a context and a list of upvars, build a closure. This just
// collects the upvars and packages them up for build_environment.
fn build_closure(cx: @block_ctxt,
                 upvars: @[ast::def],
                 mode: closure_constr_mode)
    -> {ptr: ValueRef, ptrty: ty::t, bcx: @block_ctxt} {
    // If we need to, package up the iterator body to call
    let env_vals = [];
    // Package up the upvars
    for def in *upvars {
        let lv = trans_local_var(cx, def);
        let nid = ast_util::def_id_of_def(def).node;
        let ty = ty::node_id_to_monotype(bcx_tcx(cx), nid);
        alt mode {
          for_block. { ty = ty::mk_mut_ptr(bcx_tcx(cx), ty); }
          for_send. | for_closure. {}
        }
        env_vals += [env_direct(lv.val, ty, lv.kind == owned)];
    }
    ret build_environment(cx, copy cx.fcx.lltydescs, env_vals, mode);
}

// Return a pointer to the stored typarams in a closure.
// This is awful. Since the size of the bindings stored in the closure might
// be dynamically sized, we can't skip past them to get to the tydescs until
// we have loaded the tydescs. Thus we use the stored size of the bindings
// in the tydesc for the closure to skip over them. Ugh.
fn find_environment_tydescs(bcx: @block_ctxt, envty: ty::t, closure: ValueRef)
   -> ValueRef {
    ret if !ty::type_has_dynamic_size(bcx_tcx(bcx), envty) {

            // If we can find the typarams statically, do it
            GEPi(bcx, closure,
                 [0, abi::box_rc_field_body, abi::closure_elt_ty_params])
        } else {
            // Ugh. We need to load the size of the bindings out of the
            // closure's tydesc and use that to skip over the bindings.
            let descsty =
                ty::get_element_type(bcx_tcx(bcx), envty,
                                     abi::closure_elt_ty_params as uint);
            let llenv = GEPi(bcx, closure, [0, abi::box_rc_field_body]);
            // Load the tydesc and find the size of the body
            let lldesc =
                Load(bcx, GEPi(bcx, llenv, [0, abi::closure_elt_tydesc]));
            let llsz =
                Load(bcx, GEPi(bcx, lldesc, [0, abi::tydesc_field_size]));

            // Get the bindings pointer and add the size to it
            let llbinds = GEPi(bcx, llenv, [0, abi::closure_elt_bindings]);
            bump_ptr(bcx, descsty, llbinds, llsz)
        }
}

// Given an enclosing block context, a new function context, a closure type,
// and a list of upvars, generate code to load and populate the environment
// with the upvars and type descriptors.
fn load_environment(enclosing_cx: @block_ctxt, fcx: @fn_ctxt, envty: ty::t,
                    upvars: @[ast::def], mode: closure_constr_mode) {
    let bcx = new_raw_block_ctxt(fcx, fcx.llloadenv);

    let ty = ty::mk_imm_box(bcx_tcx(bcx), envty);

    let ccx = bcx_ccx(bcx);
    let sp = bcx.sp;
    // FIXME: should have postcondition on mk_imm_box,
    // so this check won't be necessary
    check (type_has_static_size(ccx, ty));
    let llty = type_of(ccx, sp, ty);
    let llclosure = PointerCast(bcx, fcx.llenv, llty);

    // Populate the type parameters from the environment. We need to
    // do this first because the tydescs are needed to index into
    // the bindings if they are dynamically sized.
    let tydesc_count = vec::len(enclosing_cx.fcx.lltydescs);
    let lltydescs = find_environment_tydescs(bcx, envty, llclosure);
    let i = 0u;
    while i < tydesc_count {
        let lltydescptr = GEPi(bcx, lltydescs, [0, i as int]);
        fcx.lltydescs += [Load(bcx, lltydescptr)];
        i += 1u;
    }

    // Populate the upvars from the environment.
    let path = [0, abi::box_rc_field_body, abi::closure_elt_bindings];
    i = 0u;
    // Load the actual upvars.
    for upvar_def in *upvars {
        // Silly check
        check type_is_tup_like(bcx, ty);
        let upvarptr = GEP_tup_like(bcx, ty, llclosure, path + [i as int]);
        bcx = upvarptr.bcx;
        let llupvarptr = upvarptr.val;
        alt mode {
          for_block. { llupvarptr = Load(bcx, llupvarptr); }
          for_send. | for_closure. { }
        }
        let def_id = ast_util::def_id_of_def(upvar_def);
        fcx.llupvars.insert(def_id.node, llupvarptr);
        i += 1u;
    }
}

fn trans_expr_fn(bcx: @block_ctxt, f: ast::_fn, sp: span,
                 id: ast::node_id, dest: dest) -> @block_ctxt {
    if dest == ignore { ret bcx; }
    let ccx = bcx_ccx(bcx), bcx = bcx;
    let fty = node_id_type(ccx, id);
    check returns_non_ty_var(ccx, fty);
    let llfnty = type_of_fn_from_ty(ccx, sp, fty, 0u);
    let sub_cx = extend_path(bcx.fcx.lcx, ccx.names.next("anon"));
    let s = mangle_internal_name_by_path(ccx, sub_cx.path);
    let llfn = decl_internal_cdecl_fn(ccx.llmod, s, llfnty);

    let mode = alt f.proto {
      ast::proto_shared(_) { for_closure }
      ast::proto_send. { for_send }
      ast::proto_bare. | ast::proto_block. { for_block }
    };
    let env;
    alt f.proto {
      ast::proto_block. | ast::proto_shared(_) | ast::proto_send. {
        let upvars = get_freevars(ccx.tcx, id);
        let env_r = build_closure(bcx, upvars, mode);
        env = env_r.ptr;
        bcx = env_r.bcx;
        trans_closure(sub_cx, sp, f, llfn, none, [], id, {|fcx|
            load_environment(bcx, fcx, env_r.ptrty, upvars, mode);
        });
      }
      ast::proto_bare. {
        env = C_null(T_opaque_closure_ptr(ccx));
        trans_closure(sub_cx, sp, f, llfn, none, [], id, {|_fcx|});
      }
    };
    fill_fn_pair(bcx, get_dest_addr(dest), llfn, env);
    ret bcx;
}

fn trans_bind(cx: @block_ctxt, f: @ast::expr, args: [option::t<@ast::expr>],
              id: ast::node_id, dest: dest) -> @block_ctxt {
    let f_res = trans_callee(cx, f);
    ret trans_bind_1(cx, ty::expr_ty(bcx_tcx(cx), f), f_res, args,
                     ty::node_id_to_type(bcx_tcx(cx), id), dest);
}

fn trans_bind_1(cx: @block_ctxt, outgoing_fty: ty::t,
                f_res: lval_maybe_callee,
                args: [option::t<@ast::expr>], pair_ty: ty::t,
                dest: dest) -> @block_ctxt {
    let bound: [@ast::expr] = [];
    for argopt: option::t<@ast::expr> in args {
        alt argopt { none. { } some(e) { bound += [e]; } }
    }
    let bcx = f_res.bcx;
    if dest == ignore {
        for ex in bound { bcx = trans_expr(bcx, ex, ignore); }
        ret bcx;
    }

    // Figure out which tydescs we need to pass, if any.
    let outgoing_fty_real; // the type with typarams still in it
    let lltydescs: [ValueRef];
    alt f_res.generic {
      none. { outgoing_fty_real = outgoing_fty; lltydescs = []; }
      some(ginfo) {
        lazily_emit_all_generic_info_tydesc_glues(cx, ginfo);
        outgoing_fty_real = ginfo.item_type;
        lltydescs = ginfo.tydescs;
      }
    }

    let ty_param_count = vec::len(lltydescs);
    if vec::len(bound) == 0u && ty_param_count == 0u {
        // Trivial 'binding': just return the closure
        let lv = lval_maybe_callee_to_lval(f_res, pair_ty);
        bcx = lv.bcx;
        ret memmove_ty(bcx, get_dest_addr(dest), lv.val, pair_ty);
    }
    let closure = alt f_res.env {
      null_env. { none }
      _ { let (_, cl) = maybe_add_env(cx, f_res); some(cl) }
    };

    // FIXME: should follow from a precondition on trans_bind_1
    let ccx = bcx_ccx(cx);
    check (type_has_static_size(ccx, outgoing_fty));

    // Arrange for the bound function to live in the first binding spot
    // if the function is not statically known.
    let (env_vals, target_res) = alt closure {
      some(cl) {
        // Cast the function we are binding to be the type that the
        // closure will expect it to have. The type the closure knows
        // about has the type parameters substituted with the real types.
        let sp = cx.sp;
        let llclosurety = T_ptr(type_of(ccx, sp, outgoing_fty));
        let src_loc = PointerCast(bcx, cl, llclosurety);
        ([env_direct(src_loc, pair_ty, true)], none)
      }
      none. { ([], some(f_res.val)) }
    };

    // Actually construct the closure
    let closure = build_environment(bcx, lltydescs, env_vals +
                                    vec::map({|x| env_expr(x)}, bound),
                                    for_closure);
    bcx = closure.bcx;

    // Make thunk
    let llthunk =
        trans_bind_thunk(cx.fcx.lcx, cx.sp, pair_ty, outgoing_fty_real, args,
                         closure.ptrty, ty_param_count, target_res);

    // Fill the function pair
    fill_fn_pair(bcx, get_dest_addr(dest), llthunk.val, closure.ptr);
    ret bcx;
}

