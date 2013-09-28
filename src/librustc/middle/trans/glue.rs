// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//!
//
// Code relating to taking, dropping, etc as well as type descriptors.


use back::abi;
use back::link::*;
use driver::session;
use lib;
use lib::llvm::{llvm, ValueRef, True};
use middle::lang_items::{FreeFnLangItem, ExchangeFreeFnLangItem};
use middle::trans::adt;
use middle::trans::base::*;
use middle::trans::callee;
use middle::trans::closure;
use middle::trans::common::*;
use middle::trans::build::*;
use middle::trans::expr;
use middle::trans::machine::*;
use middle::trans::reflect;
use middle::trans::tvec;
use middle::trans::type_of::type_of;
use middle::trans::uniq;
use middle::ty;
use util::ppaux;
use util::ppaux::ty_to_short_str;

use middle::trans::type_::Type;

use std::c_str::ToCStr;
use std::libc::c_uint;
use syntax::ast;

pub fn trans_free(cx: @mut Block, v: ValueRef) -> @mut Block {
    let _icx = push_ctxt("trans_free");
    callee::trans_lang_call(cx,
        langcall(cx, None, "", FreeFnLangItem),
        [PointerCast(cx, v, Type::i8p())],
        Some(expr::Ignore)).bcx
}

pub fn trans_exchange_free(cx: @mut Block, v: ValueRef) -> @mut Block {
    let _icx = push_ctxt("trans_exchange_free");
    callee::trans_lang_call(cx,
        langcall(cx, None, "", ExchangeFreeFnLangItem),
        [PointerCast(cx, v, Type::i8p())],
        Some(expr::Ignore)).bcx
}

pub fn take_ty(cx: @mut Block, v: ValueRef, t: ty::t) -> @mut Block {
    // NB: v is an *alias* of type t here, not a direct value.
    let _icx = push_ctxt("take_ty");
    if ty::type_needs_drop(cx.tcx(), t) {
        return call_tydesc_glue(cx, v, t, abi::tydesc_field_take_glue);
    }
    return cx;
}

pub fn drop_ty(cx: @mut Block, v: ValueRef, t: ty::t) -> @mut Block {
    // NB: v is an *alias* of type t here, not a direct value.
    let _icx = push_ctxt("drop_ty");
    if ty::type_needs_drop(cx.tcx(), t) {
        return call_tydesc_glue(cx, v, t, abi::tydesc_field_drop_glue);
    }
    return cx;
}

pub fn drop_ty_immediate(bcx: @mut Block, v: ValueRef, t: ty::t) -> @mut Block {
    let _icx = push_ctxt("drop_ty_immediate");
    match ty::get(t).sty {
        ty::ty_uniq(_)
      | ty::ty_evec(_, ty::vstore_uniq)
      | ty::ty_estr(ty::vstore_uniq) => {
        free_ty_immediate(bcx, v, t)
      }
        ty::ty_box(_) | ty::ty_opaque_box
      | ty::ty_evec(_, ty::vstore_box)
      | ty::ty_estr(ty::vstore_box) => {
        decr_refcnt_maybe_free(bcx, v, None, t)
      }
      _ => bcx.tcx().sess.bug("drop_ty_immediate: non-box ty")
    }
}

pub fn free_ty(cx: @mut Block, v: ValueRef, t: ty::t) -> @mut Block {
    // NB: v is an *alias* of type t here, not a direct value.
    let _icx = push_ctxt("free_ty");
    if ty::type_needs_drop(cx.tcx(), t) {
        return call_tydesc_glue(cx, v, t, abi::tydesc_field_free_glue);
    }
    return cx;
}

pub fn free_ty_immediate(bcx: @mut Block, v: ValueRef, t: ty::t) -> @mut Block {
    let _icx = push_ctxt("free_ty_immediate");
    match ty::get(t).sty {
      ty::ty_uniq(_) |
      ty::ty_evec(_, ty::vstore_uniq) |
      ty::ty_estr(ty::vstore_uniq) |
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_evec(_, ty::vstore_box) |
      ty::ty_estr(ty::vstore_box) |
      ty::ty_opaque_closure_ptr(_) => {
        let vp = alloca(bcx, type_of(bcx.ccx(), t), "");
        Store(bcx, v, vp);
        free_ty(bcx, vp, t)
      }
      _ => bcx.tcx().sess.bug("free_ty_immediate: non-box ty")
    }
}

pub fn lazily_emit_all_tydesc_glue(ccx: @mut CrateContext,
                                   static_ti: @mut tydesc_info) {
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_take_glue, static_ti);
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_drop_glue, static_ti);
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_free_glue, static_ti);
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_visit_glue, static_ti);
}

pub fn simplified_glue_type(tcx: ty::ctxt, field: uint, t: ty::t) -> ty::t {
    if (field == abi::tydesc_field_take_glue ||
        field == abi::tydesc_field_drop_glue ||
        field == abi::tydesc_field_free_glue) &&
        ! ty::type_needs_drop(tcx, t) {
          return ty::mk_u32();
    }

    if field == abi::tydesc_field_take_glue {
        match ty::get(t).sty {
          ty::ty_unboxed_vec(*) |
              ty::ty_uniq(*) |
              ty::ty_estr(ty::vstore_uniq) |
              ty::ty_evec(_, ty::vstore_uniq) => { return ty::mk_u32(); }
          _ => ()
        }
    }

    if field == abi::tydesc_field_take_glue &&
        ty::type_is_boxed(t) {
          return ty::mk_imm_box(tcx, ty::mk_u32());
    }

    if field == abi::tydesc_field_free_glue {
        match ty::get(t).sty {
          ty::ty_bare_fn(*) |
          ty::ty_closure(*) |
          ty::ty_box(*) |
          ty::ty_opaque_box |
          ty::ty_uniq(*) |
          ty::ty_evec(_, ty::vstore_uniq) | ty::ty_estr(ty::vstore_uniq) |
          ty::ty_evec(_, ty::vstore_box) | ty::ty_estr(ty::vstore_box) |
          ty::ty_opaque_closure_ptr(*) => (),
          _ => { return ty::mk_u32(); }
        }
    }

    if (field == abi::tydesc_field_free_glue ||
        field == abi::tydesc_field_drop_glue) {
        match ty::get(t).sty {
          ty::ty_box(mt) |
          ty::ty_evec(mt, ty::vstore_box)
          if ! ty::type_needs_drop(tcx, mt.ty) =>
          return ty::mk_imm_box(tcx, ty::mk_u32()),

          ty::ty_uniq(mt) |
          ty::ty_evec(mt, ty::vstore_uniq)
          if ! ty::type_needs_drop(tcx, mt.ty) =>
          return ty::mk_imm_uniq(tcx, ty::mk_u32()),

          _ => ()
        }
    }

    return t;
}

pub fn lazily_emit_simplified_tydesc_glue(ccx: @mut CrateContext,
                                          field: uint,
                                          ti: &mut tydesc_info) -> bool {
    let _icx = push_ctxt("lazily_emit_simplified_tydesc_glue");
    let simpl = simplified_glue_type(ccx.tcx, field, ti.ty);
    if simpl != ti.ty {
        let simpl_ti = get_tydesc(ccx, simpl);
        lazily_emit_tydesc_glue(ccx, field, simpl_ti);
        {
            if field == abi::tydesc_field_take_glue {
                ti.take_glue = simpl_ti.take_glue;
            } else if field == abi::tydesc_field_drop_glue {
                ti.drop_glue = simpl_ti.drop_glue;
            } else if field == abi::tydesc_field_free_glue {
                ti.free_glue = simpl_ti.free_glue;
            } else if field == abi::tydesc_field_visit_glue {
                ti.visit_glue = simpl_ti.visit_glue;
            }
        }
        return true;
    }
    return false;
}


pub fn lazily_emit_tydesc_glue(ccx: @mut CrateContext,
                               field: uint,
                               ti: @mut tydesc_info) {
    let _icx = push_ctxt("lazily_emit_tydesc_glue");
    let llfnty = Type::glue_fn(type_of::type_of(ccx, ti.ty).ptr_to());

    if lazily_emit_simplified_tydesc_glue(ccx, field, ti) {
        return;
    }

    if field == abi::tydesc_field_take_glue {
        match ti.take_glue {
          Some(_) => (),
          None => {
            debug!("+++ lazily_emit_tydesc_glue TAKE %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, "take");
            ti.take_glue = Some(glue_fn);
            make_generic_glue(ccx, ti.ty, glue_fn, make_take_glue, "take");
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
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, "drop");
            ti.drop_glue = Some(glue_fn);
            make_generic_glue(ccx, ti.ty, glue_fn, make_drop_glue, "drop");
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
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, "free");
            ti.free_glue = Some(glue_fn);
            make_generic_glue(ccx, ti.ty, glue_fn, make_free_glue, "free");
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
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, "visit");
            ti.visit_glue = Some(glue_fn);
            make_generic_glue(ccx, ti.ty, glue_fn, make_visit_glue, "visit");
            debug!("--- lazily_emit_tydesc_glue VISIT %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
          }
        }
    }
}

// See [Note-arg-mode]
pub fn call_tydesc_glue_full(bcx: @mut Block,
                             v: ValueRef,
                             tydesc: ValueRef,
                             field: uint,
                             static_ti: Option<@mut tydesc_info>) {
    let _icx = push_ctxt("call_tydesc_glue_full");
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

    // When static type info is available, avoid casting parameter unless the
    // glue is using a simplified type, because the function already has the
    // right type. Otherwise cast to generic pointer.
    let llrawptr = if static_ti.is_none() || static_glue_fn.is_none() {
        PointerCast(bcx, v, Type::i8p())
    } else {
        let ty = static_ti.unwrap().ty;
        let simpl = simplified_glue_type(ccx.tcx, field, ty);
        if simpl != ty {
            PointerCast(bcx, v, type_of(ccx, simpl).ptr_to())
        } else {
            v
        }
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

    Call(bcx, llfn, [C_null(Type::nil().ptr_to()), llrawptr], []);
}

// See [Note-arg-mode]
pub fn call_tydesc_glue(cx: @mut Block, v: ValueRef, t: ty::t, field: uint)
    -> @mut Block {
    let _icx = push_ctxt("call_tydesc_glue");
    let ti = get_tydesc(cx.ccx(), t);
    call_tydesc_glue_full(cx, v, ti.tydesc, field, Some(ti));
    return cx;
}

pub fn make_visit_glue(bcx: @mut Block, v: ValueRef, t: ty::t) -> @mut Block {
    let _icx = push_ctxt("make_visit_glue");
    do with_scope(bcx, None, "visitor cleanup") |bcx| {
        let mut bcx = bcx;
        let (visitor_trait, object_ty) = match ty::visitor_object_ty(bcx.tcx(),
                                                                     ty::re_static) {
            Ok(pair) => pair,
            Err(s) => {
                bcx.tcx().sess.fatal(s);
            }
        };
        let v = PointerCast(bcx, v, type_of::type_of(bcx.ccx(), object_ty).ptr_to());
        bcx = reflect::emit_calls_to_trait_visit_ty(bcx, t, v, visitor_trait.def_id);
        // The visitor is a boxed object and needs to be dropped
        add_clean(bcx, v, object_ty);
        bcx
    }
}

pub fn make_free_glue(bcx: @mut Block, v: ValueRef, t: ty::t) -> @mut Block {
    // NB: v0 is an *alias* of type t here, not a direct value.
    let _icx = push_ctxt("make_free_glue");
    match ty::get(t).sty {
      ty::ty_box(body_mt) => {
        let v = Load(bcx, v);
        let body = GEPi(bcx, v, [0u, abi::box_field_body]);
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
        make_free_glue(bcx, v, tvec::expand_boxed_vec_ty(bcx.tcx(), t))
      }
      ty::ty_closure(_) => {
        closure::make_closure_glue(bcx, v, t, free_ty)
      }
      ty::ty_opaque_closure_ptr(ck) => {
        closure::make_opaque_cbox_free_glue(bcx, ck, v)
      }
      _ => bcx
    }
}

pub fn trans_struct_drop_flag(bcx: @mut Block, t: ty::t, v0: ValueRef, dtor_did: ast::DefId,
                              class_did: ast::DefId, substs: &ty::substs) -> @mut Block {
    let repr = adt::represent_type(bcx.ccx(), t);
    let drop_flag = adt::trans_drop_flag_ptr(bcx, repr, v0);
    do with_cond(bcx, IsNotNull(bcx, Load(bcx, drop_flag))) |cx| {
        let mut bcx = cx;

        // Find and call the actual destructor
        let dtor_addr = get_res_dtor(bcx.ccx(), dtor_did,
                                     class_did, substs.tps.clone());

        // The second argument is the "self" argument for drop
        let params = unsafe {
            let ty = Type::from_ref(llvm::LLVMTypeOf(dtor_addr));
            ty.element_type().func_params()
        };

        // Class dtors have no explicit args, so the params should
        // just consist of the environment (self)
        assert_eq!(params.len(), 1);

        let self_arg = PointerCast(bcx, v0, params[0]);
        let args = ~[self_arg];

        Call(bcx, dtor_addr, args, []);

        // Drop the fields
        let field_tys = ty::struct_fields(bcx.tcx(), class_did, substs);
        for (i, fld) in field_tys.iter().enumerate() {
            let llfld_a = adt::trans_field_ptr(bcx, repr, v0, 0, i);
            bcx = drop_ty(bcx, llfld_a, fld.mt.ty);
        }

        Store(bcx, C_u8(0), drop_flag);
        bcx
    }
}

pub fn trans_struct_drop(mut bcx: @mut Block, t: ty::t, v0: ValueRef, dtor_did: ast::DefId,
                         class_did: ast::DefId, substs: &ty::substs) -> @mut Block {
    let repr = adt::represent_type(bcx.ccx(), t);

    // Find and call the actual destructor
    let dtor_addr = get_res_dtor(bcx.ccx(), dtor_did,
                                 class_did, substs.tps.clone());

    // The second argument is the "self" argument for drop
    let params = unsafe {
        let ty = Type::from_ref(llvm::LLVMTypeOf(dtor_addr));
        ty.element_type().func_params()
    };

    // Class dtors have no explicit args, so the params should
    // just consist of the environment (self)
    assert_eq!(params.len(), 1);

    let self_arg = PointerCast(bcx, v0, params[0]);
    let args = ~[self_arg];

    Call(bcx, dtor_addr, args, []);

    // Drop the fields
    let field_tys = ty::struct_fields(bcx.tcx(), class_did, substs);
    for (i, fld) in field_tys.iter().enumerate() {
        let llfld_a = adt::trans_field_ptr(bcx, repr, v0, 0, i);
        bcx = drop_ty(bcx, llfld_a, fld.mt.ty);
    }

    bcx
}

pub fn make_drop_glue(bcx: @mut Block, v0: ValueRef, t: ty::t) -> @mut Block {
    // NB: v0 is an *alias* of type t here, not a direct value.
    let _icx = push_ctxt("make_drop_glue");
    let ccx = bcx.ccx();
    match ty::get(t).sty {
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_estr(ty::vstore_box) | ty::ty_evec(_, ty::vstore_box) => {
        decr_refcnt_maybe_free(bcx, Load(bcx, v0), Some(v0), t)
      }
      ty::ty_uniq(_) |
      ty::ty_evec(_, ty::vstore_uniq) | ty::ty_estr(ty::vstore_uniq) => {
        free_ty(bcx, v0, t)
      }
      ty::ty_unboxed_vec(_) => {
        tvec::make_drop_glue_unboxed(bcx, v0, t)
      }
      ty::ty_struct(did, ref substs) => {
        let tcx = bcx.tcx();
        match ty::ty_dtor(tcx, did) {
          ty::TraitDtor(dtor, true) => {
            trans_struct_drop_flag(bcx, t, v0, dtor, did, substs)
          }
          ty::TraitDtor(dtor, false) => {
            trans_struct_drop(bcx, t, v0, dtor, did, substs)
          }
          ty::NoDtor => {
            // No dtor? Just the default case
            iter_structural_ty(bcx, v0, t, drop_ty)
          }
        }
      }
      ty::ty_closure(_) => {
        closure::make_closure_glue(bcx, v0, t, drop_ty)
      }
      ty::ty_trait(_, _, ty::BoxTraitStore, _, _) => {
          let llbox_ptr = GEPi(bcx, v0, [0u, abi::trt_field_box]);
          let llbox = Load(bcx, llbox_ptr);
          decr_refcnt_maybe_free(bcx, llbox, Some(llbox_ptr),
                                 ty::mk_opaque_box(ccx.tcx))
      }
      ty::ty_trait(_, _, ty::UniqTraitStore, _, _) => {
          let lluniquevalue = GEPi(bcx, v0, [0, abi::trt_field_box]);
          // Only drop the value when it is non-null
          do with_cond(bcx, IsNotNull(bcx, Load(bcx, lluniquevalue))) |bcx| {
              let llvtable = Load(bcx, GEPi(bcx, v0, [0, abi::trt_field_vtable]));

              // Cast the vtable to a pointer to a pointer to a tydesc.
              let llvtable = PointerCast(bcx, llvtable,
                                         ccx.tydesc_type.ptr_to().ptr_to());
              let lltydesc = Load(bcx, llvtable);
              call_tydesc_glue_full(bcx,
                                    lluniquevalue,
                                    lltydesc,
                                    abi::tydesc_field_free_glue,
                                    None);
              bcx
          }
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
    }
}

// box_ptr_ptr is optional, it is constructed if not supplied.
pub fn decr_refcnt_maybe_free(bcx: @mut Block, box_ptr: ValueRef,
                              box_ptr_ptr: Option<ValueRef>,
                              t: ty::t)
                           -> @mut Block {
    let _icx = push_ctxt("decr_refcnt_maybe_free");
    let ccx = bcx.ccx();

    let decr_bcx = sub_block(bcx, "decr");
    let free_bcx = sub_block(decr_bcx, "free");
    let next_bcx = sub_block(bcx, "next");
    CondBr(bcx, IsNotNull(bcx, box_ptr), decr_bcx.llbb, next_bcx.llbb);

    let rc_ptr = GEPi(decr_bcx, box_ptr, [0u, abi::box_field_refcnt]);
    let rc = Sub(decr_bcx, Load(decr_bcx, rc_ptr), C_int(ccx, 1));
    Store(decr_bcx, rc, rc_ptr);
    CondBr(decr_bcx, IsNull(decr_bcx, rc), free_bcx.llbb, next_bcx.llbb);

    let free_bcx = match box_ptr_ptr {
        Some(p) => free_ty(free_bcx, p, t),
        None => free_ty_immediate(free_bcx, box_ptr, t)
    };
    Br(free_bcx, next_bcx.llbb);

    next_bcx
}


pub fn make_take_glue(bcx: @mut Block, v: ValueRef, t: ty::t) -> @mut Block {
    let _icx = push_ctxt("make_take_glue");
    // NB: v is a *pointer* to type t here, not a direct value.
    match ty::get(t).sty {
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_evec(_, ty::vstore_box) | ty::ty_estr(ty::vstore_box) => {
        incr_refcnt_of_boxed(bcx, Load(bcx, v)); bcx
      }
      ty::ty_evec(_, ty::vstore_slice(_))
      | ty::ty_estr(ty::vstore_slice(_)) => {
        bcx
      }
      ty::ty_closure(_) => bcx,
      ty::ty_trait(_, _, ty::BoxTraitStore, _, _) => {
        let llbox = Load(bcx, GEPi(bcx, v, [0u, abi::trt_field_box]));
        incr_refcnt_of_boxed(bcx, llbox);
        bcx
      }
      ty::ty_trait(_, _, ty::UniqTraitStore, _, _) => {
          let lluniquevalue = GEPi(bcx, v, [0, abi::trt_field_box]);
          let llvtable = Load(bcx, GEPi(bcx, v, [0, abi::trt_field_vtable]));

          // Cast the vtable to a pointer to a pointer to a tydesc.
          let llvtable = PointerCast(bcx, llvtable,
                                     bcx.ccx().tydesc_type.ptr_to().ptr_to());
          let lltydesc = Load(bcx, llvtable);
          call_tydesc_glue_full(bcx,
                                lluniquevalue,
                                lltydesc,
                                abi::tydesc_field_take_glue,
                                None);
          bcx
      }
      ty::ty_opaque_closure_ptr(_) => bcx,
      ty::ty_struct(did, _) => {
        let tcx = bcx.tcx();
        let bcx = iter_structural_ty(bcx, v, t, take_ty);

        match ty::ty_dtor(tcx, did) {
          ty::TraitDtor(_, false) => {
            // Zero out the struct
            unsafe {
                let ty = Type::from_ref(llvm::LLVMTypeOf(v));
                memzero(&B(bcx), v, ty);
            }

          }
          _ => { }
        }
        bcx
      }
      _ if ty::type_is_structural(t) => {
        iter_structural_ty(bcx, v, t, take_ty)
      }
      _ => bcx
    }
}

pub fn incr_refcnt_of_boxed(cx: @mut Block, box_ptr: ValueRef) {
    let _icx = push_ctxt("incr_refcnt_of_boxed");
    let ccx = cx.ccx();
    let rc_ptr = GEPi(cx, box_ptr, [0u, abi::box_field_refcnt]);
    let rc = Load(cx, rc_ptr);
    let rc = Add(cx, rc, C_int(ccx, 1));
    Store(cx, rc, rc_ptr);
}


// Generates the declaration for (but doesn't emit) a type descriptor.
pub fn declare_tydesc(ccx: &mut CrateContext, t: ty::t) -> @mut tydesc_info {
    // If emit_tydescs already ran, then we shouldn't be creating any new
    // tydescs.
    assert!(!ccx.finished_tydescs);

    let llty = type_of(ccx, t);

    if ccx.sess.count_type_sizes() {
        println!("{}\t{}", llsize_of_real(ccx, llty),
                 ppaux::ty_to_str(ccx.tcx, t));
    }

    let has_header = match ty::get(t).sty {
        ty::ty_box(*) => true,
        ty::ty_uniq(*) => ty::type_contents(ccx.tcx, t).contains_managed(),
        _ => false
    };

    let borrow_offset = if has_header {
        ccx.offsetof_gep(llty, [0u, abi::box_field_body])
    } else {
        C_uint(ccx, 0)
    };

    let llsize = llsize_of(ccx, llty);
    let llalign = llalign_of(ccx, llty);
    let name = mangle_internal_name_by_type_and_seq(ccx, t, "tydesc").to_managed();
    note_unique_llvm_symbol(ccx, name);
    debug!("+++ declare_tydesc %s %s", ppaux::ty_to_str(ccx.tcx, t), name);
    let gvar = do name.with_c_str |buf| {
        unsafe {
            llvm::LLVMAddGlobal(ccx.llmod, ccx.tydesc_type.to_ref(), buf)
        }
    };

    let ty_name = C_estr_slice(ccx, ppaux::ty_to_str(ccx.tcx, t).to_managed());

    let inf = @mut tydesc_info {
        ty: t,
        tydesc: gvar,
        size: llsize,
        align: llalign,
        borrow_offset: borrow_offset,
        name: ty_name,
        take_glue: None,
        drop_glue: None,
        free_glue: None,
        visit_glue: None
    };
    debug!("--- declare_tydesc %s", ppaux::ty_to_str(ccx.tcx, t));
    return inf;
}

pub type glue_helper<'self> = &'self fn(@mut Block, ValueRef, ty::t) -> @mut Block;

pub fn declare_generic_glue(ccx: &mut CrateContext, t: ty::t, llfnty: Type,
                            name: &str) -> ValueRef {
    let _icx = push_ctxt("declare_generic_glue");
    let fn_nm = mangle_internal_name_by_type_and_seq(ccx, t, (~"glue_" + name)).to_managed();
    debug!("%s is for type %s", fn_nm, ppaux::ty_to_str(ccx.tcx, t));
    note_unique_llvm_symbol(ccx, fn_nm);
    let llfn = decl_cdecl_fn(ccx.llmod, fn_nm, llfnty);
    set_glue_inlining(llfn, t);
    return llfn;
}

pub fn make_generic_glue_inner(ccx: @mut CrateContext,
                               t: ty::t,
                               llfn: ValueRef,
                               helper: glue_helper)
                            -> ValueRef {
    let _icx = push_ctxt("make_generic_glue_inner");
    let fcx = new_fn_ctxt(ccx, ~[], llfn, ty::mk_nil(), None);
    lib::llvm::SetLinkage(llfn, lib::llvm::InternalLinkage);
    ccx.stats.n_glues_created += 1u;
    // All glue functions take values passed *by alias*; this is a
    // requirement since in many contexts glue is invoked indirectly and
    // the caller has no idea if it's dealing with something that can be
    // passed by value.
    //
    // llfn is expected be declared to take a parameter of the appropriate
    // type, so we don't need to explicitly cast the function parameter.

    let bcx = fcx.entry_bcx.unwrap();
    let rawptr0_arg = fcx.arg_pos(0u);
    let llrawptr0 = unsafe { llvm::LLVMGetParam(llfn, rawptr0_arg as c_uint) };
    let bcx = helper(bcx, llrawptr0, t);

    finish_fn(fcx, bcx);

    return llfn;
}

pub fn make_generic_glue(ccx: @mut CrateContext,
                         t: ty::t,
                         llfn: ValueRef,
                         helper: glue_helper,
                         name: &str)
                      -> ValueRef {
    let _icx = push_ctxt("make_generic_glue");
    let glue_name = fmt!("glue %s %s", name, ty_to_short_str(ccx.tcx, t));
    let _s = StatRecorder::new(ccx, glue_name);
    make_generic_glue_inner(ccx, t, llfn, helper)
}

pub fn emit_tydescs(ccx: &mut CrateContext) {
    let _icx = push_ctxt("emit_tydescs");
    // As of this point, allow no more tydescs to be created.
    ccx.finished_tydescs = true;
    let glue_fn_ty = Type::generic_glue_fn(ccx).ptr_to();
    let tyds = &mut ccx.tydescs;
    for (_, &val) in tyds.iter() {
        let ti = val;

        // Each of the glue functions needs to be cast to a generic type
        // before being put into the tydesc because we only have a singleton
        // tydesc type. Then we'll recast each function to its real type when
        // calling it.
        let take_glue =
            match ti.take_glue {
              None => { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              Some(v) => {
                unsafe {
                    ccx.stats.n_real_glues += 1u;
                    llvm::LLVMConstPointerCast(v, glue_fn_ty.to_ref())
                }
              }
            };
        let drop_glue =
            match ti.drop_glue {
              None => { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              Some(v) => {
                unsafe {
                    ccx.stats.n_real_glues += 1u;
                    llvm::LLVMConstPointerCast(v, glue_fn_ty.to_ref())
                }
              }
            };
        let free_glue =
            match ti.free_glue {
              None => { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              Some(v) => {
                unsafe {
                    ccx.stats.n_real_glues += 1u;
                    llvm::LLVMConstPointerCast(v, glue_fn_ty.to_ref())
                }
              }
            };
        let visit_glue =
            match ti.visit_glue {
              None => { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              Some(v) => {
                unsafe {
                    ccx.stats.n_real_glues += 1u;
                    llvm::LLVMConstPointerCast(v, glue_fn_ty.to_ref())
                }
              }
            };

        debug!("ti.borrow_offset: %s",
               ccx.tn.val_to_str(ti.borrow_offset));

        let tydesc = C_named_struct(ccx.tydesc_type,
                                    [ti.size, // size
                                     ti.align, // align
                                     take_glue, // take_glue
                                     drop_glue, // drop_glue
                                     free_glue, // free_glue
                                     visit_glue, // visit_glue
                                     ti.borrow_offset, // borrow_offset
                                     ti.name]); // name

        unsafe {
            let gvar = ti.tydesc;
            llvm::LLVMSetInitializer(gvar, tydesc);
            llvm::LLVMSetGlobalConstant(gvar, True);
            lib::llvm::SetLinkage(gvar, lib::llvm::InternalLinkage);
        }
    };
}
