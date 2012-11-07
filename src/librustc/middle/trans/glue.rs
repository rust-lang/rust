//!
//
// Code relating to taking, dropping, etc as well as type descriptors.

use lib::llvm::{ValueRef, TypeRef};
use base::*;
use common::*;
use build::*;
use type_of::type_of;

fn trans_free(cx: block, v: ValueRef) -> block {
    let _icx = cx.insn_ctxt("trans_free");
    callee::trans_rtcall(cx, ~"free", ~[PointerCast(cx, v, T_ptr(T_i8()))],
                         expr::Ignore)
}

fn trans_unique_free(cx: block, v: ValueRef) -> block {
    let _icx = cx.insn_ctxt("trans_unique_free");
    callee::trans_rtcall(
        cx, ~"exchange_free", ~[PointerCast(cx, v, T_ptr(T_i8()))],
        expr::Ignore)
}

fn take_ty(cx: block, v: ValueRef, t: ty::t) -> block {
    // NB: v is an *alias* of type t here, not a direct value.
    let _icx = cx.insn_ctxt("take_ty");
    if ty::type_needs_drop(cx.tcx(), t) {
        return call_tydesc_glue(cx, v, t, abi::tydesc_field_take_glue);
    }
    return cx;
}

fn drop_ty(cx: block, v: ValueRef, t: ty::t) -> block {
    // NB: v is an *alias* of type t here, not a direct value.
    let _icx = cx.insn_ctxt("drop_ty");
    if ty::type_needs_drop(cx.tcx(), t) {
        return call_tydesc_glue(cx, v, t, abi::tydesc_field_drop_glue);
    }
    return cx;
}

fn drop_ty_root(bcx: block, v: ValueRef, rooted: bool, t: ty::t) -> block {
    if rooted {
        // NB: v is a raw ptr to an addrspace'd ptr to the value.
        let v = PointerCast(bcx, Load(bcx, v), T_ptr(type_of(bcx.ccx(), t)));
        drop_ty(bcx, v, t)
    } else {
        drop_ty(bcx, v, t)
    }
}

fn drop_ty_immediate(bcx: block, v: ValueRef, t: ty::t) -> block {
    let _icx = bcx.insn_ctxt("drop_ty_immediate");
    match ty::get(t).sty {
      ty::ty_uniq(_) |
      ty::ty_evec(_, ty::vstore_uniq) |
      ty::ty_estr(ty::vstore_uniq) => {
        free_ty_immediate(bcx, v, t)
      }
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_evec(_, ty::vstore_box) |
      ty::ty_estr(ty::vstore_box) => {
        decr_refcnt_maybe_free(bcx, v, t)
      }
      _ => bcx.tcx().sess.bug(~"drop_ty_immediate: non-box ty")
    }
}

fn take_ty_immediate(bcx: block, v: ValueRef, t: ty::t) -> Result {
    let _icx = bcx.insn_ctxt("take_ty_immediate");
    match ty::get(t).sty {
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_evec(_, ty::vstore_box) |
      ty::ty_estr(ty::vstore_box) => {
        incr_refcnt_of_boxed(bcx, v);
        rslt(bcx, v)
      }
      ty::ty_uniq(_) => {
        uniq::duplicate(bcx, v, t)
      }
      ty::ty_evec(_, ty::vstore_uniq) |
      ty::ty_estr(ty::vstore_uniq) => {
        tvec::duplicate_uniq(bcx, v, t)
      }
      _ => rslt(bcx, v)
    }
}

fn free_ty(cx: block, v: ValueRef, t: ty::t) -> block {
    // NB: v is an *alias* of type t here, not a direct value.
    let _icx = cx.insn_ctxt("free_ty");
    if ty::type_needs_drop(cx.tcx(), t) {
        return call_tydesc_glue(cx, v, t, abi::tydesc_field_free_glue);
    }
    return cx;
}

fn free_ty_immediate(bcx: block, v: ValueRef, t: ty::t) -> block {
    let _icx = bcx.insn_ctxt("free_ty_immediate");
    match ty::get(t).sty {
      ty::ty_uniq(_) |
      ty::ty_evec(_, ty::vstore_uniq) |
      ty::ty_estr(ty::vstore_uniq) |
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_evec(_, ty::vstore_box) |
      ty::ty_estr(ty::vstore_box) |
      ty::ty_opaque_closure_ptr(_) => {
        let vp = alloca_zeroed(bcx, type_of(bcx.ccx(), t));
        Store(bcx, v, vp);
        free_ty(bcx, vp, t)
      }
      _ => bcx.tcx().sess.bug(~"free_ty_immediate: non-box ty")
    }
}

fn lazily_emit_all_tydesc_glue(ccx: @crate_ctxt,
                               static_ti: @tydesc_info) {
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_take_glue, static_ti);
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_drop_glue, static_ti);
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_free_glue, static_ti);
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_visit_glue, static_ti);
}

fn simplified_glue_type(tcx: ty::ctxt, field: uint, t: ty::t) -> ty::t {

    if (field == abi::tydesc_field_take_glue ||
        field == abi::tydesc_field_drop_glue ||
        field == abi::tydesc_field_free_glue) &&
        ! ty::type_needs_drop(tcx, t) {
          return ty::mk_u32(tcx);
    }

    if field == abi::tydesc_field_take_glue {
        match ty::get(t).sty {
          ty::ty_unboxed_vec(*) => { return ty::mk_u32(tcx); }
          _ => ()
        }
    }

    if field == abi::tydesc_field_take_glue &&
        ty::type_is_boxed(t) {
          return ty::mk_imm_box(tcx, ty::mk_u32(tcx));
    }

    if field == abi::tydesc_field_free_glue {
        match ty::get(t).sty {
          ty::ty_fn(*) |
          ty::ty_box(*) |
          ty::ty_opaque_box |
          ty::ty_uniq(*) |
          ty::ty_evec(_, ty::vstore_uniq) | ty::ty_estr(ty::vstore_uniq) |
          ty::ty_evec(_, ty::vstore_box) | ty::ty_estr(ty::vstore_box) |
          ty::ty_opaque_closure_ptr(*) => (),
          _ => { return ty::mk_u32(tcx); }
        }
    }

    if (field == abi::tydesc_field_free_glue ||
        field == abi::tydesc_field_drop_glue) {
        match ty::get(t).sty {
          ty::ty_box(mt) |
          ty::ty_evec(mt, ty::vstore_box)
          if ! ty::type_needs_drop(tcx, mt.ty) =>
          return ty::mk_imm_box(tcx, ty::mk_u32(tcx)),

          ty::ty_uniq(mt) |
          ty::ty_evec(mt, ty::vstore_uniq)
          if ! ty::type_needs_drop(tcx, mt.ty) =>
          return ty::mk_imm_uniq(tcx, ty::mk_u32(tcx)),

          _ => ()
        }
    }

    return t;
}

pure fn cast_glue(ccx: @crate_ctxt, ti: @tydesc_info, v: ValueRef)
    -> ValueRef {
    unsafe {
        let llfnty = type_of_glue_fn(ccx, ti.ty);
        llvm::LLVMConstPointerCast(v, T_ptr(llfnty))
    }
}

fn lazily_emit_simplified_tydesc_glue(ccx: @crate_ctxt, field: uint,
                                      ti: @tydesc_info) -> bool {
    let _icx = ccx.insn_ctxt("lazily_emit_simplified_tydesc_glue");
    let simpl = simplified_glue_type(ccx.tcx, field, ti.ty);
    if simpl != ti.ty {
        let simpl_ti = base::get_tydesc(ccx, simpl);
        lazily_emit_tydesc_glue(ccx, field, simpl_ti);
        if field == abi::tydesc_field_take_glue {
            ti.take_glue =
                simpl_ti.take_glue.map(|v| cast_glue(ccx, ti, *v));
        } else if field == abi::tydesc_field_drop_glue {
            ti.drop_glue =
                simpl_ti.drop_glue.map(|v| cast_glue(ccx, ti, *v));
        } else if field == abi::tydesc_field_free_glue {
            ti.free_glue =
                simpl_ti.free_glue.map(|v| cast_glue(ccx, ti, *v));
        } else if field == abi::tydesc_field_visit_glue {
            ti.visit_glue =
                simpl_ti.visit_glue.map(|v| cast_glue(ccx, ti, *v));
        }
        return true;
    }
    return false;
}


fn lazily_emit_tydesc_glue(ccx: @crate_ctxt, field: uint,
                           ti: @tydesc_info) {
    let _icx = ccx.insn_ctxt("lazily_emit_tydesc_glue");
    let llfnty = type_of_glue_fn(ccx, ti.ty);

    if lazily_emit_simplified_tydesc_glue(ccx, field, ti) {
        return;
    }

    if field == abi::tydesc_field_take_glue {
        match ti.take_glue {
          Some(_) => (),
          None => {
            debug!("+++ lazily_emit_tydesc_glue TAKE %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, ~"take");
            ti.take_glue = Some(glue_fn);
            make_generic_glue(ccx, ti.ty, glue_fn, make_take_glue, ~"take");
            debug!("--- lazily_emit_tydesc_glue TAKE %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
          }
        }
    } else if field == abi::tydesc_field_drop_glue {
        match ti.drop_glue {
          Some(_) => (),
          None => {
            debug!("+++ lazily_emit_tydesc_glue DROP %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, ~"drop");
            ti.drop_glue = Some(glue_fn);
            make_generic_glue(ccx, ti.ty, glue_fn, make_drop_glue, ~"drop");
            debug!("--- lazily_emit_tydesc_glue DROP %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
          }
        }
    } else if field == abi::tydesc_field_free_glue {
        match ti.free_glue {
          Some(_) => (),
          None => {
            debug!("+++ lazily_emit_tydesc_glue FREE %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, ~"free");
            ti.free_glue = Some(glue_fn);
            make_generic_glue(ccx, ti.ty, glue_fn, make_free_glue, ~"free");
            debug!("--- lazily_emit_tydesc_glue FREE %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
          }
        }
    } else if field == abi::tydesc_field_visit_glue {
        match ti.visit_glue {
          Some(_) => (),
          None => {
            debug!("+++ lazily_emit_tydesc_glue VISIT %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, ~"visit");
            ti.visit_glue = Some(glue_fn);
            make_generic_glue(ccx, ti.ty, glue_fn, make_visit_glue, ~"visit");
            debug!("--- lazily_emit_tydesc_glue VISIT %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
          }
        }
    }
}

// See [Note-arg-mode]
fn call_tydesc_glue_full(++bcx: block, v: ValueRef, tydesc: ValueRef,
                         field: uint, static_ti: Option<@tydesc_info>) {
    let _icx = bcx.insn_ctxt("call_tydesc_glue_full");
    let ccx = bcx.ccx();
    // NB: Don't short-circuit even if this block is unreachable because
    // GC-based cleanup needs to the see that the roots are live.
    let no_lpads =
        ccx.sess.opts.debugging_opts & session::no_landing_pads != 0;
    if bcx.unreachable && !no_lpads { return; }

    let static_glue_fn = match static_ti {
      None => None,
      Some(sti) => {
        lazily_emit_tydesc_glue(ccx, field, sti);
        if field == abi::tydesc_field_take_glue {
            sti.take_glue
        } else if field == abi::tydesc_field_drop_glue {
            sti.drop_glue
        } else if field == abi::tydesc_field_free_glue {
            sti.free_glue
        } else if field == abi::tydesc_field_visit_glue {
            sti.visit_glue
        } else {
            None
        }
      }
    };

    // When available, use static type info to give glue the right type.
    let static_glue_fn = match static_ti {
      None => None,
      Some(sti) => {
        match static_glue_fn {
          None => None,
          Some(sgf) => Some(
              PointerCast(bcx, sgf, T_ptr(type_of_glue_fn(ccx, sti.ty))))
        }
      }
    };

    // When static type info is available, avoid casting parameter because the
    // function already has the right type. Otherwise cast to generic pointer.
    let llrawptr = if static_ti.is_none() || static_glue_fn.is_none() {
        PointerCast(bcx, v, T_ptr(T_i8()))
    } else {
        v
    };

    let llfn = {
        match static_glue_fn {
          None => {
            // Select out the glue function to call from the tydesc
            let llfnptr = GEPi(bcx, tydesc, [0u, field]);
            Load(bcx, llfnptr)
          }
          Some(sgf) => sgf
        }
    };

    Call(bcx, llfn, ~[C_null(T_ptr(T_nil())), C_null(T_ptr(T_nil())),
                      C_null(T_ptr(T_ptr(bcx.ccx().tydesc_type))), llrawptr]);
}

// See [Note-arg-mode]
fn call_tydesc_glue(++cx: block, v: ValueRef, t: ty::t, field: uint)
    -> block {
    let _icx = cx.insn_ctxt("call_tydesc_glue");
    let ti = get_tydesc(cx.ccx(), t);
    call_tydesc_glue_full(cx, v, ti.tydesc, field, Some(ti));
    return cx;
}

fn make_visit_glue(bcx: block, v: ValueRef, t: ty::t) {
    let _icx = bcx.insn_ctxt("make_visit_glue");
    let mut bcx = bcx;
    let ty_visitor_name = special_idents::ty_visitor;
    assert bcx.ccx().tcx.intrinsic_defs.contains_key(ty_visitor_name);
    let (trait_id, ty) = bcx.ccx().tcx.intrinsic_defs.get(ty_visitor_name);
    let v = PointerCast(bcx, v, T_ptr(type_of::type_of(bcx.ccx(), ty)));
    bcx = reflect::emit_calls_to_trait_visit_ty(bcx, t, v, trait_id);
    build_return(bcx);
}

fn make_free_glue(bcx: block, v: ValueRef, t: ty::t) {
    // NB: v0 is an *alias* of type t here, not a direct value.
    let _icx = bcx.insn_ctxt("make_free_glue");
    let ccx = bcx.ccx();
    let bcx = match ty::get(t).sty {
      ty::ty_box(body_mt) => {
        let v = Load(bcx, v);
        let body = GEPi(bcx, v, [0u, abi::box_field_body]);
        // Cast away the addrspace of the box pointer.
        let body = PointerCast(bcx, body, T_ptr(type_of(ccx, body_mt.ty)));
        let bcx = drop_ty(bcx, body, body_mt.ty);
        trans_free(bcx, v)
      }
      ty::ty_opaque_box => {
        let v = Load(bcx, v);
        let td = Load(bcx, GEPi(bcx, v, [0u, abi::box_field_tydesc]));
        let valptr = GEPi(bcx, v, [0u, abi::box_field_body]);
        // Generate code that, dynamically, indexes into the
        // tydesc and calls the drop glue that got set dynamically
        call_tydesc_glue_full(bcx, valptr, td, abi::tydesc_field_drop_glue,
                              None);
        trans_free(bcx, v)
      }
      ty::ty_uniq(*) => {
        uniq::make_free_glue(bcx, v, t)
      }
      ty::ty_evec(_, ty::vstore_uniq) | ty::ty_estr(ty::vstore_uniq) |
      ty::ty_evec(_, ty::vstore_box) | ty::ty_estr(ty::vstore_box) => {
        make_free_glue(bcx, v,
                       tvec::expand_boxed_vec_ty(bcx.tcx(), t));
        return;
      }
      ty::ty_fn(_) => {
        closure::make_fn_glue(bcx, v, t, free_ty)
      }
      ty::ty_opaque_closure_ptr(ck) => {
        closure::make_opaque_cbox_free_glue(bcx, ck, v)
      }
      ty::ty_class(did, ref substs) => {
        // Call the dtor if there is one
        do option::map_default(&ty::ty_dtor(bcx.tcx(), did), bcx) |dt_id| {
            trans_class_drop(bcx, v, *dt_id, did, substs)
        }
      }
      _ => bcx
    };
    build_return(bcx);
}

fn trans_class_drop(bcx: block,
                    v0: ValueRef,
                    dtor_did: ast::def_id,
                    class_did: ast::def_id,
                    substs: &ty::substs) -> block {
    let drop_flag = GEPi(bcx, v0, struct_dtor());
    do with_cond(bcx, IsNotNull(bcx, Load(bcx, drop_flag))) |cx| {
        let mut bcx = cx;

        // Find and call the actual destructor
        let dtor_addr = get_res_dtor(bcx.ccx(), dtor_did,
                                     class_did, substs.tps);

        // The second argument is the "self" argument for drop
        let params = lib::llvm::fn_ty_param_tys(
            llvm::LLVMGetElementType(llvm::LLVMTypeOf(dtor_addr)));

        // Class dtors have no explicit args, so the params should
        // just consist of the output pointer and the environment
        // (self)
        assert(params.len() == 2);
        let self_arg = PointerCast(bcx, v0, params[1]);
        let args = ~[bcx.fcx.llretptr, self_arg];
        Call(bcx, dtor_addr, args);

        // Drop the fields
        let field_tys =
            ty::class_items_as_mutable_fields(bcx.tcx(), class_did,
                                              substs);
        for vec::eachi(field_tys) |i, fld| {
            let llfld_a = GEPi(bcx, v0, struct_field(i));
            bcx = drop_ty(bcx, llfld_a, fld.mt.ty);
        }

        Store(bcx, C_u8(0), drop_flag);
        bcx
    }
}


fn make_drop_glue(bcx: block, v0: ValueRef, t: ty::t) {
    // NB: v0 is an *alias* of type t here, not a direct value.
    let _icx = bcx.insn_ctxt("make_drop_glue");
    let ccx = bcx.ccx();
    let bcx = match ty::get(t).sty {
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_estr(ty::vstore_box) | ty::ty_evec(_, ty::vstore_box) => {
        decr_refcnt_maybe_free(bcx, Load(bcx, v0), t)
      }
      ty::ty_uniq(_) |
      ty::ty_evec(_, ty::vstore_uniq) | ty::ty_estr(ty::vstore_uniq) => {
        free_ty(bcx, v0, t)
      }
      ty::ty_unboxed_vec(_) => {
        tvec::make_drop_glue_unboxed(bcx, v0, t)
      }
      ty::ty_class(did, ref substs) => {
        let tcx = bcx.tcx();
        match ty::ty_dtor(tcx, did) {
          Some(dtor) => {
            trans_class_drop(bcx, v0, dtor, did, substs)
          }
          None => {
            // No dtor? Just the default case
            iter_structural_ty(bcx, v0, t, drop_ty)
          }
        }
      }
      ty::ty_fn(_) => {
        closure::make_fn_glue(bcx, v0, t, drop_ty)
      }
      ty::ty_trait(_, _, ty::vstore_box) => {
        let llbox = Load(bcx, GEPi(bcx, v0, [0u, 1u]));
        decr_refcnt_maybe_free(bcx, llbox, ty::mk_opaque_box(ccx.tcx))
      }
      ty::ty_trait(_, _, ty::vstore_uniq) => {
        let lluniquevalue = GEPi(bcx, v0, [0, 1]);
        let lltydesc = Load(bcx, GEPi(bcx, v0, [0, 2]));
        call_tydesc_glue_full(bcx, lluniquevalue, lltydesc,
                              abi::tydesc_field_free_glue, None);
        bcx
      }
      ty::ty_opaque_closure_ptr(ck) => {
        closure::make_opaque_cbox_drop_glue(bcx, ck, v0)
      }
      _ => {
        if ty::type_needs_drop(ccx.tcx, t) &&
            ty::type_is_structural(t) {
            iter_structural_ty(bcx, v0, t, drop_ty)
        } else { bcx }
      }
    };
    build_return(bcx);
}

fn decr_refcnt_maybe_free(bcx: block, box_ptr: ValueRef, t: ty::t) -> block {
    let _icx = bcx.insn_ctxt("decr_refcnt_maybe_free");
    let ccx = bcx.ccx();

    do with_cond(bcx, IsNotNull(bcx, box_ptr)) |bcx| {
        let rc_ptr = GEPi(bcx, box_ptr, [0u, abi::box_field_refcnt]);
        let rc = Sub(bcx, Load(bcx, rc_ptr), C_int(ccx, 1));
        Store(bcx, rc, rc_ptr);
        let zero_test = ICmp(bcx, lib::llvm::IntEQ, C_int(ccx, 0), rc);
        with_cond(bcx, zero_test, |bcx| free_ty_immediate(bcx, box_ptr, t))
    }
}


fn make_take_glue(bcx: block, v: ValueRef, t: ty::t) {
    let _icx = bcx.insn_ctxt("make_take_glue");
    // NB: v is a *pointer* to type t here, not a direct value.
    let bcx = match ty::get(t).sty {
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_evec(_, ty::vstore_box) | ty::ty_estr(ty::vstore_box) => {
        incr_refcnt_of_boxed(bcx, Load(bcx, v)); bcx
      }
      ty::ty_uniq(_) => {
        let Result {bcx, val} = uniq::duplicate(bcx, Load(bcx, v), t);
        Store(bcx, val, v);
        bcx
      }
      ty::ty_evec(_, ty::vstore_uniq) | ty::ty_estr(ty::vstore_uniq) => {
        let Result {bcx, val} = tvec::duplicate_uniq(bcx, Load(bcx, v), t);
        Store(bcx, val, v);
        bcx
      }
      ty::ty_evec(_, ty::vstore_slice(_))
      | ty::ty_estr(ty::vstore_slice(_)) => {
        bcx
      }
      ty::ty_fn(_) => {
        closure::make_fn_glue(bcx, v, t, take_ty)
      }
      ty::ty_trait(_, _, ty::vstore_box) => {
        let llbox = Load(bcx, GEPi(bcx, v, [0u, 1u]));
        incr_refcnt_of_boxed(bcx, llbox);
        bcx
      }
      ty::ty_trait(_, _, ty::vstore_uniq) => {
        let llval = GEPi(bcx, v, [0, 1]);
        let lltydesc = Load(bcx, GEPi(bcx, v, [0, 2]));
        call_tydesc_glue_full(bcx, llval, lltydesc,
                              abi::tydesc_field_take_glue, None);
        bcx
      }
      ty::ty_opaque_closure_ptr(ck) => {
        closure::make_opaque_cbox_take_glue(bcx, ck, v)
      }
      _ if ty::type_is_structural(t) => {
        iter_structural_ty(bcx, v, t, take_ty)
      }
      _ => bcx
    };

    build_return(bcx);
}

fn incr_refcnt_of_boxed(cx: block, box_ptr: ValueRef) {
    let _icx = cx.insn_ctxt("incr_refcnt_of_boxed");
    let ccx = cx.ccx();
    let rc_ptr = GEPi(cx, box_ptr, [0u, abi::box_field_refcnt]);
    let rc = Load(cx, rc_ptr);
    let rc = Add(cx, rc, C_int(ccx, 1));
    Store(cx, rc, rc_ptr);
}


// Chooses the addrspace for newly declared types.
fn declare_tydesc_addrspace(ccx: @crate_ctxt, t: ty::t) -> addrspace {
    if !ty::type_needs_drop(ccx.tcx, t) {
        return default_addrspace;
    } else if ty::type_is_immediate(t) {
        // For immediate types, we don't actually need an addrspace, because
        // e.g. boxed types include pointers to their contents which are
        // already correctly tagged with addrspaces.
        return default_addrspace;
    } else {
        return ccx.next_addrspace();
    }
}

// Generates the declaration for (but doesn't emit) a type descriptor.
fn declare_tydesc(ccx: @crate_ctxt, t: ty::t) -> @tydesc_info {
    let _icx = ccx.insn_ctxt("declare_tydesc");
    // If emit_tydescs already ran, then we shouldn't be creating any new
    // tydescs.
    assert !ccx.finished_tydescs;

    let llty = type_of(ccx, t);

    if ccx.sess.count_type_sizes() {
        io::println(fmt!("%u\t%s",
                         llsize_of_real(ccx, llty),
                         ty_to_str(ccx.tcx, t)));
    }

    let llsize = llsize_of(ccx, llty);
    let llalign = llalign_of(ccx, llty);
    let addrspace = declare_tydesc_addrspace(ccx, t);
    //XXX this triggers duplicate LLVM symbols
    let name = if false /*ccx.sess.opts.debuginfo*/ {
        mangle_internal_name_by_type_only(ccx, t, ~"tydesc")
    } else {
        mangle_internal_name_by_seq(ccx, ~"tydesc")
    };
    note_unique_llvm_symbol(ccx, name);
    log(debug, fmt!("+++ declare_tydesc %s %s", ty_to_str(ccx.tcx, t), name));
    let gvar = str::as_c_str(name, |buf| {
        llvm::LLVMAddGlobal(ccx.llmod, ccx.tydesc_type, buf)
    });
    let inf =
        @{ty: t,
          tydesc: gvar,
          size: llsize,
          align: llalign,
          addrspace: addrspace,
          mut take_glue: None,
          mut drop_glue: None,
          mut free_glue: None,
          mut visit_glue: None};
    log(debug, ~"--- declare_tydesc " + ppaux::ty_to_str(ccx.tcx, t));
    return inf;
}

type glue_helper = fn@(block, ValueRef, ty::t);

fn declare_generic_glue(ccx: @crate_ctxt, t: ty::t, llfnty: TypeRef,
                        name: ~str) -> ValueRef {
    let _icx = ccx.insn_ctxt("declare_generic_glue");
    let name = name;
    let mut fn_nm;
    //XXX this triggers duplicate LLVM symbols
    if false /*ccx.sess.opts.debuginfo*/ {
        fn_nm = mangle_internal_name_by_type_only(ccx, t, (~"glue_" + name));
    } else {
        fn_nm = mangle_internal_name_by_seq(ccx, (~"glue_" + name));
    }
    debug!("%s is for type %s", fn_nm, ty_to_str(ccx.tcx, t));
    note_unique_llvm_symbol(ccx, fn_nm);
    let llfn = decl_cdecl_fn(ccx.llmod, fn_nm, llfnty);
    set_glue_inlining(llfn, t);
    return llfn;
}

fn make_generic_glue_inner(ccx: @crate_ctxt, t: ty::t,
                           llfn: ValueRef, helper: glue_helper) -> ValueRef {
    let _icx = ccx.insn_ctxt("make_generic_glue_inner");
    let fcx = new_fn_ctxt(ccx, ~[], llfn, None);
    lib::llvm::SetLinkage(llfn, lib::llvm::InternalLinkage);
    ccx.stats.n_glues_created += 1u;
    // All glue functions take values passed *by alias*; this is a
    // requirement since in many contexts glue is invoked indirectly and
    // the caller has no idea if it's dealing with something that can be
    // passed by value.
    //
    // llfn is expected be declared to take a parameter of the appropriate
    // type, so we don't need to explicitly cast the function parameter.

    let bcx = top_scope_block(fcx, None);
    let lltop = bcx.llbb;
    let llrawptr0 = llvm::LLVMGetParam(llfn, 3u as c_uint);
    helper(bcx, llrawptr0, t);
    finish_fn(fcx, lltop);
    return llfn;
}

fn make_generic_glue(ccx: @crate_ctxt, t: ty::t, llfn: ValueRef,
                     helper: glue_helper, name: ~str)
    -> ValueRef {
    let _icx = ccx.insn_ctxt("make_generic_glue");
    if !ccx.sess.trans_stats() {
        return make_generic_glue_inner(ccx, t, llfn, helper);
    }

    let start = time::get_time();
    let llval = make_generic_glue_inner(ccx, t, llfn, helper);
    let end = time::get_time();
    log_fn_time(ccx, ~"glue " + name + ~" " + ty_to_short_str(ccx.tcx, t),
                start, end);
    return llval;
}

fn emit_tydescs(ccx: @crate_ctxt) {
    let _icx = ccx.insn_ctxt("emit_tydescs");
    // As of this point, allow no more tydescs to be created.
    ccx.finished_tydescs = true;
    for ccx.tydescs.each |_key, val| {
        let glue_fn_ty = T_ptr(T_generic_glue_fn(ccx));
        let ti = val;

        // Each of the glue functions needs to be cast to a generic type
        // before being put into the tydesc because we only have a singleton
        // tydesc type. Then we'll recast each function to its real type when
        // calling it.
        let take_glue =
            match copy ti.take_glue {
              None => { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              Some(v) => {
                ccx.stats.n_real_glues += 1u;
                llvm::LLVMConstPointerCast(v, glue_fn_ty)
              }
            };
        let drop_glue =
            match copy ti.drop_glue {
              None => { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              Some(v) => {
                ccx.stats.n_real_glues += 1u;
                llvm::LLVMConstPointerCast(v, glue_fn_ty)
              }
            };
        let free_glue =
            match copy ti.free_glue {
              None => { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              Some(v) => {
                ccx.stats.n_real_glues += 1u;
                llvm::LLVMConstPointerCast(v, glue_fn_ty)
              }
            };
        let visit_glue =
            match copy ti.visit_glue {
              None => { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              Some(v) => {
                ccx.stats.n_real_glues += 1u;
                llvm::LLVMConstPointerCast(v, glue_fn_ty)
              }
            };

        let shape = C_null(T_ptr(T_i8()));
        let shape_tables = C_null(T_ptr(T_i8()));

        let tydesc =
            C_named_struct(ccx.tydesc_type,
                           ~[ti.size, // size
                             ti.align, // align
                             take_glue, // take_glue
                             drop_glue, // drop_glue
                             free_glue, // free_glue
                             visit_glue, // visit_glue
                             shape, // shape
                             shape_tables]); // shape_tables

        let gvar = ti.tydesc;
        llvm::LLVMSetInitializer(gvar, tydesc);
        llvm::LLVMSetGlobalConstant(gvar, True);
        lib::llvm::SetLinkage(gvar, lib::llvm::InternalLinkage);

        // Index tydesc by addrspace.
        if ti.addrspace > gc_box_addrspace {
            let llty = T_ptr(ccx.tydesc_type);
            let addrspace_name = fmt!("_gc_addrspace_metadata_%u",
                                      ti.addrspace as uint);
            let addrspace_gvar = str::as_c_str(addrspace_name, |buf| {
                llvm::LLVMAddGlobal(ccx.llmod, llty, buf)
            });
            lib::llvm::SetLinkage(addrspace_gvar, lib::llvm::InternalLinkage);
            llvm::LLVMSetInitializer(addrspace_gvar, gvar);
        }
    };
}
