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
use lib::llvm::{llvm, ValueRef, True};
use lib;
use middle::lang_items::{FreeFnLangItem, ExchangeFreeFnLangItem};
use middle::trans::adt;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::cleanup;
use middle::trans::cleanup::CleanupMethods;
use middle::trans::common::*;
use middle::trans::expr;
use middle::trans::machine::*;
use middle::trans::reflect;
use middle::trans::tvec;
use middle::trans::type_::Type;
use middle::trans::type_of::{type_of, sizing_type_of};
use middle::ty;
use util::ppaux::ty_to_short_str;
use util::ppaux;

use arena::TypedArena;
use std::c_str::ToCStr;
use std::cell::Cell;
use libc::c_uint;
use syntax::ast;
use syntax::parse::token;

pub fn trans_free<'a>(cx: &'a Block<'a>, v: ValueRef) -> &'a Block<'a> {
    let _icx = push_ctxt("trans_free");
    callee::trans_lang_call(cx,
        langcall(cx, None, "", FreeFnLangItem),
        [PointerCast(cx, v, Type::i8p(cx.ccx()))],
        Some(expr::Ignore)).bcx
}

pub fn trans_exchange_free<'a>(cx: &'a Block<'a>, v: ValueRef)
                           -> &'a Block<'a> {
    let _icx = push_ctxt("trans_exchange_free");
    callee::trans_lang_call(cx,
        langcall(cx, None, "", ExchangeFreeFnLangItem),
        [PointerCast(cx, v, Type::i8p(cx.ccx()))],
        Some(expr::Ignore)).bcx
}

pub fn take_ty<'a>(bcx: &'a Block<'a>, v: ValueRef, t: ty::t)
               -> &'a Block<'a> {
    // NB: v is an *alias* of type t here, not a direct value.
    let _icx = push_ctxt("take_ty");
    match ty::get(t).sty {
        ty::ty_box(_) => incr_refcnt_of_boxed(bcx, v),
        _ if ty::type_is_structural(t)
          && ty::type_needs_drop(bcx.tcx(), t) => {
            iter_structural_ty(bcx, v, t, take_ty)
        }
        _ => bcx
    }
}

fn get_drop_glue_type(ccx: &CrateContext, t: ty::t) -> ty::t {
    let tcx = ccx.tcx();
    if !ty::type_needs_drop(tcx, t) {
        return ty::mk_i8();
    }
    match ty::get(t).sty {
        ty::ty_box(typ) if !ty::type_needs_drop(tcx, typ) =>
            ty::mk_box(tcx, ty::mk_i8()),

        ty::ty_uniq(typ) if !ty::type_needs_drop(tcx, typ) => {
            let llty = sizing_type_of(ccx, typ);
            // Unique boxes do not allocate for zero-size types. The standard library may assume
            // that `free` is never called on the pointer returned for `~ZeroSizeType`.
            if llsize_of_alloc(ccx, llty) == 0 {
                ty::mk_i8()
            } else {
                ty::mk_uniq(tcx, ty::mk_i8())
            }
        }

        ty::ty_vec(ty, ty::VstoreUniq) if !ty::type_needs_drop(tcx, ty) =>
            ty::mk_uniq(tcx, ty::mk_i8()),

        _ => t
    }
}

pub fn drop_ty<'a>(bcx: &'a Block<'a>, v: ValueRef, t: ty::t)
               -> &'a Block<'a> {
    // NB: v is an *alias* of type t here, not a direct value.
    let _icx = push_ctxt("drop_ty");
    let ccx = bcx.ccx();
    if ty::type_needs_drop(bcx.tcx(), t) {
        let glue = get_drop_glue(ccx, t);
        let glue_type = get_drop_glue_type(ccx, t);
        let ptr = if glue_type != t {
            PointerCast(bcx, v, type_of(ccx, glue_type).ptr_to())
        } else {
            v
        };
        Call(bcx, glue, [ptr], []);
    }
    bcx
}

pub fn drop_ty_immediate<'a>(bcx: &'a Block<'a>, v: ValueRef, t: ty::t)
                         -> &'a Block<'a> {
    let _icx = push_ctxt("drop_ty_immediate");
    let vp = alloca(bcx, type_of(bcx.ccx(), t), "");
    Store(bcx, v, vp);
    drop_ty(bcx, vp, t)
}

pub fn get_drop_glue(ccx: &CrateContext, t: ty::t) -> ValueRef {
    let t = get_drop_glue_type(ccx, t);
    match ccx.drop_glues.borrow().find(&t) {
        Some(&glue) => return glue,
        _ => { }
    }

    let llfnty = Type::glue_fn(ccx, type_of(ccx, t).ptr_to());
    let glue = declare_generic_glue(ccx, t, llfnty, "drop");

    ccx.drop_glues.borrow_mut().insert(t, glue);

    make_generic_glue(ccx, t, glue, make_drop_glue, "drop");

    glue
}

pub fn lazily_emit_visit_glue(ccx: &CrateContext, ti: @tydesc_info) {
    let _icx = push_ctxt("lazily_emit_visit_glue");

    let llfnty = Type::glue_fn(ccx, type_of(ccx, ti.ty).ptr_to());

    match ti.visit_glue.get() {
        Some(_) => (),
        None => {
            debug!("+++ lazily_emit_tydesc_glue VISIT {}", ppaux::ty_to_str(ccx.tcx(), ti.ty));
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, "visit");
            ti.visit_glue.set(Some(glue_fn));
            make_generic_glue(ccx, ti.ty, glue_fn, make_visit_glue, "visit");
            debug!("--- lazily_emit_tydesc_glue VISIT {}", ppaux::ty_to_str(ccx.tcx(), ti.ty));
        }
    }
}

// See [Note-arg-mode]
pub fn call_visit_glue(bcx: &Block, v: ValueRef, tydesc: ValueRef,
                       static_ti: Option<@tydesc_info>) {
    let _icx = push_ctxt("call_tydesc_glue_full");
    let ccx = bcx.ccx();
    // NB: Don't short-circuit even if this block is unreachable because
    // GC-based cleanup needs to the see that the roots are live.
    if bcx.unreachable.get() && !ccx.sess().no_landing_pads() { return; }

    let static_glue_fn = match static_ti {
        None => None,
        Some(sti) => {
            lazily_emit_visit_glue(ccx, sti);
            sti.visit_glue.get()
        }
    };

    // When static type info is available, avoid casting to a generic pointer.
    let llrawptr = if static_glue_fn.is_none() {
        PointerCast(bcx, v, Type::i8p(ccx))
    } else {
        v
    };

    let llfn = {
        match static_glue_fn {
            None => {
                // Select out the glue function to call from the tydesc
                let llfnptr = GEPi(bcx, tydesc, [0u, abi::tydesc_field_visit_glue]);
                Load(bcx, llfnptr)
            }
            Some(sgf) => sgf
        }
    };

    Call(bcx, llfn, [llrawptr], []);
}

fn make_visit_glue<'a>(bcx: &'a Block<'a>, v: ValueRef, t: ty::t)
                   -> &'a Block<'a> {
    let _icx = push_ctxt("make_visit_glue");
    let mut bcx = bcx;
    let (visitor_trait, object_ty) = match ty::visitor_object_ty(bcx.tcx(),
                                                                 ty::ReStatic) {
        Ok(pair) => pair,
        Err(s) => {
            bcx.tcx().sess.fatal(s);
        }
    };
    let v = PointerCast(bcx, v, type_of(bcx.ccx(), object_ty).ptr_to());
    bcx = reflect::emit_calls_to_trait_visit_ty(bcx, t, v, visitor_trait.def_id);
    bcx
}

fn trans_struct_drop_flag<'a>(bcx: &'a Block<'a>,
                              t: ty::t,
                              v0: ValueRef,
                              dtor_did: ast::DefId,
                              class_did: ast::DefId,
                              substs: &ty::substs)
                              -> &'a Block<'a> {
    let repr = adt::represent_type(bcx.ccx(), t);
    let drop_flag = adt::trans_drop_flag_ptr(bcx, repr, v0);
    with_cond(bcx, IsNotNull(bcx, Load(bcx, drop_flag)), |cx| {
        trans_struct_drop(cx, t, v0, dtor_did, class_did, substs)
    })
}

fn trans_struct_drop<'a>(bcx: &'a Block<'a>,
                         t: ty::t,
                         v0: ValueRef,
                         dtor_did: ast::DefId,
                         class_did: ast::DefId,
                         substs: &ty::substs)
                         -> &'a Block<'a> {
    let repr = adt::represent_type(bcx.ccx(), t);

    // Find and call the actual destructor
    let dtor_addr = get_res_dtor(bcx.ccx(), dtor_did,
                                 class_did, substs.tps.as_slice());

    // The second argument is the "self" argument for drop
    let params = unsafe {
        let ty = Type::from_ref(llvm::LLVMTypeOf(dtor_addr));
        ty.element_type().func_params()
    };

    // Class dtors have no explicit args, so the params should
    // just consist of the environment (self)
    assert_eq!(params.len(), 1);

    // Be sure to put all of the fields into a scope so we can use an invoke
    // instruction to call the user destructor but still call the field
    // destructors if the user destructor fails.
    let field_scope = bcx.fcx.push_custom_cleanup_scope();

    let self_arg = PointerCast(bcx, v0, *params.get(0));
    let args = vec!(self_arg);

    // Add all the fields as a value which needs to be cleaned at the end of
    // this scope.
    let field_tys = ty::struct_fields(bcx.tcx(), class_did, substs);
    for (i, fld) in field_tys.iter().enumerate() {
        let llfld_a = adt::trans_field_ptr(bcx, repr, v0, 0, i);
        bcx.fcx.schedule_drop_mem(cleanup::CustomScope(field_scope),
                                  llfld_a,
                                  fld.mt.ty);
    }

    let (_, bcx) = invoke(bcx, dtor_addr, args, [], None);

    bcx.fcx.pop_and_trans_custom_cleanup_scope(bcx, field_scope)
}

fn make_drop_glue<'a>(bcx: &'a Block<'a>, v0: ValueRef, t: ty::t) -> &'a Block<'a> {
    // NB: v0 is an *alias* of type t here, not a direct value.
    let _icx = push_ctxt("make_drop_glue");
    match ty::get(t).sty {
        ty::ty_box(body_ty) => {
            decr_refcnt_maybe_free(bcx, v0, body_ty)
        }
        ty::ty_uniq(content_ty) => {
            let llbox = Load(bcx, v0);
            let not_null = IsNotNull(bcx, llbox);
            with_cond(bcx, not_null, |bcx| {
                let bcx = drop_ty(bcx, llbox, content_ty);
                trans_exchange_free(bcx, llbox)
            })
        }
        ty::ty_vec(_, ty::VstoreUniq) | ty::ty_str(ty::VstoreUniq) => {
            let llbox = Load(bcx, v0);
            let not_null = IsNotNull(bcx, llbox);
            with_cond(bcx, not_null, |bcx| {
                let unit_ty = ty::sequence_element_type(bcx.tcx(), t);
                let bcx = tvec::make_drop_glue_unboxed(bcx, llbox, unit_ty);
                trans_exchange_free(bcx, llbox)
            })
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
        ty::ty_trait(~ty::TyTrait { store: ty::UniqTraitStore, .. }) => {
            let lluniquevalue = GEPi(bcx, v0, [0, abi::trt_field_box]);
            // Only drop the value when it is non-null
            with_cond(bcx, IsNotNull(bcx, Load(bcx, lluniquevalue)), |bcx| {
                let dtor_ptr = Load(bcx, GEPi(bcx, v0, [0, abi::trt_field_vtable]));
                let dtor = Load(bcx, dtor_ptr);
                Call(bcx, dtor, [PointerCast(bcx, lluniquevalue, Type::i8p(bcx.ccx()))], []);
                bcx
            })
        }
        ty::ty_closure(ref f) if f.sigil == ast::OwnedSigil => {
            let box_cell_v = GEPi(bcx, v0, [0u, abi::fn_field_box]);
            let env = Load(bcx, box_cell_v);
            let env_ptr_ty = Type::at_box(bcx.ccx(), Type::i8(bcx.ccx())).ptr_to();
            let env = PointerCast(bcx, env, env_ptr_ty);
            with_cond(bcx, IsNotNull(bcx, env), |bcx| {
                let dtor_ptr = GEPi(bcx, env, [0u, abi::box_field_tydesc]);
                let dtor = Load(bcx, dtor_ptr);
                let cdata = GEPi(bcx, env, [0u, abi::box_field_body]);
                Call(bcx, dtor, [PointerCast(bcx, cdata, Type::i8p(bcx.ccx()))], []);

                // Free the environment itself
                trans_exchange_free(bcx, env)
            })
        }
        _ => {
            if ty::type_needs_drop(bcx.tcx(), t) &&
                ty::type_is_structural(t) {
                iter_structural_ty(bcx, v0, t, drop_ty)
            } else {
                bcx
            }
        }
    }
}

fn decr_refcnt_maybe_free<'a>(bcx: &'a Block<'a>,
                              box_ptr_ptr: ValueRef,
                              t: ty::t) -> &'a Block<'a> {
    let _icx = push_ctxt("decr_refcnt_maybe_free");
    let fcx = bcx.fcx;
    let ccx = bcx.ccx();

    let decr_bcx = fcx.new_temp_block("decr");
    let free_bcx = fcx.new_temp_block("free");
    let next_bcx = fcx.new_temp_block("next");

    let box_ptr = Load(bcx, box_ptr_ptr);
    let llnotnull = IsNotNull(bcx, box_ptr);
    CondBr(bcx, llnotnull, decr_bcx.llbb, next_bcx.llbb);

    let rc_ptr = GEPi(decr_bcx, box_ptr, [0u, abi::box_field_refcnt]);
    let rc = Sub(decr_bcx, Load(decr_bcx, rc_ptr), C_int(ccx, 1));
    Store(decr_bcx, rc, rc_ptr);
    CondBr(decr_bcx, IsNull(decr_bcx, rc), free_bcx.llbb, next_bcx.llbb);

    let v = Load(free_bcx, box_ptr_ptr);
    let body = GEPi(free_bcx, v, [0u, abi::box_field_body]);
    let free_bcx = drop_ty(free_bcx, body, t);
    let free_bcx = trans_free(free_bcx, v);
    Br(free_bcx, next_bcx.llbb);

    next_bcx
}

fn incr_refcnt_of_boxed<'a>(bcx: &'a Block<'a>,
                            box_ptr_ptr: ValueRef) -> &'a Block<'a> {
    let _icx = push_ctxt("incr_refcnt_of_boxed");
    let ccx = bcx.ccx();
    let box_ptr = Load(bcx, box_ptr_ptr);
    let rc_ptr = GEPi(bcx, box_ptr, [0u, abi::box_field_refcnt]);
    let rc = Load(bcx, rc_ptr);
    let rc = Add(bcx, rc, C_int(ccx, 1));
    Store(bcx, rc, rc_ptr);
    bcx
}


// Generates the declaration for (but doesn't emit) a type descriptor.
pub fn declare_tydesc(ccx: &CrateContext, t: ty::t) -> @tydesc_info {
    // If emit_tydescs already ran, then we shouldn't be creating any new
    // tydescs.
    assert!(!ccx.finished_tydescs.get());

    let llty = type_of(ccx, t);

    if ccx.sess().count_type_sizes() {
        println!("{}\t{}", llsize_of_real(ccx, llty),
                 ppaux::ty_to_str(ccx.tcx(), t));
    }

    let llsize = llsize_of(ccx, llty);
    let llalign = llalign_of(ccx, llty);
    let name = mangle_internal_name_by_type_and_seq(ccx, t, "tydesc");
    debug!("+++ declare_tydesc {} {}", ppaux::ty_to_str(ccx.tcx(), t), name);
    let gvar = name.with_c_str(|buf| {
        unsafe {
            llvm::LLVMAddGlobal(ccx.llmod, ccx.tydesc_type().to_ref(), buf)
        }
    });
    note_unique_llvm_symbol(ccx, name);

    let ty_name = token::intern_and_get_ident(ppaux::ty_to_str(ccx.tcx(), t));
    let ty_name = C_str_slice(ccx, ty_name);

    let inf = @tydesc_info {
        ty: t,
        tydesc: gvar,
        size: llsize,
        align: llalign,
        name: ty_name,
        visit_glue: Cell::new(None),
    };
    debug!("--- declare_tydesc {}", ppaux::ty_to_str(ccx.tcx(), t));
    return inf;
}

fn declare_generic_glue(ccx: &CrateContext, t: ty::t, llfnty: Type,
                        name: &str) -> ValueRef {
    let _icx = push_ctxt("declare_generic_glue");
    let fn_nm = mangle_internal_name_by_type_and_seq(ccx, t, ~"glue_" + name);
    debug!("{} is for type {}", fn_nm, ppaux::ty_to_str(ccx.tcx(), t));
    let llfn = decl_cdecl_fn(ccx.llmod, fn_nm, llfnty, ty::mk_nil());
    note_unique_llvm_symbol(ccx, fn_nm);
    return llfn;
}

fn make_generic_glue(ccx: &CrateContext,
                     t: ty::t,
                     llfn: ValueRef,
                     helper: <'a> |&'a Block<'a>, ValueRef, ty::t|
                                  -> &'a Block<'a>,
                     name: &str)
                     -> ValueRef {
    let _icx = push_ctxt("make_generic_glue");
    let glue_name = format!("glue {} {}", name, ty_to_short_str(ccx.tcx(), t));
    let _s = StatRecorder::new(ccx, glue_name);

    let arena = TypedArena::new();
    let fcx = new_fn_ctxt(ccx, llfn, -1, false, ty::mk_nil(), None, None, &arena);

    init_function(&fcx, false, ty::mk_nil(), None);

    lib::llvm::SetLinkage(llfn, lib::llvm::InternalLinkage);
    ccx.stats.n_glues_created.set(ccx.stats.n_glues_created.get() + 1u);
    // All glue functions take values passed *by alias*; this is a
    // requirement since in many contexts glue is invoked indirectly and
    // the caller has no idea if it's dealing with something that can be
    // passed by value.
    //
    // llfn is expected be declared to take a parameter of the appropriate
    // type, so we don't need to explicitly cast the function parameter.

    let bcx = fcx.entry_bcx.borrow().clone().unwrap();
    let llrawptr0 = unsafe { llvm::LLVMGetParam(llfn, fcx.arg_pos(0) as c_uint) };
    let bcx = helper(bcx, llrawptr0, t);
    finish_fn(&fcx, bcx);

    llfn
}

pub fn emit_tydescs(ccx: &CrateContext) {
    let _icx = push_ctxt("emit_tydescs");
    // As of this point, allow no more tydescs to be created.
    ccx.finished_tydescs.set(true);
    let glue_fn_ty = Type::generic_glue_fn(ccx).ptr_to();
    for (_, &val) in ccx.tydescs.borrow().iter() {
        let ti = val;

        // Each of the glue functions needs to be cast to a generic type
        // before being put into the tydesc because we only have a singleton
        // tydesc type. Then we'll recast each function to its real type when
        // calling it.
        let drop_glue = unsafe {
            llvm::LLVMConstPointerCast(get_drop_glue(ccx, ti.ty), glue_fn_ty.to_ref())
        };
        ccx.stats.n_real_glues.set(ccx.stats.n_real_glues.get() + 1);
        let visit_glue =
            match ti.visit_glue.get() {
              None => {
                  ccx.stats.n_null_glues.set(ccx.stats.n_null_glues.get() +
                                             1u);
                  C_null(glue_fn_ty)
              }
              Some(v) => {
                unsafe {
                    ccx.stats.n_real_glues.set(ccx.stats.n_real_glues.get() +
                                               1);
                    llvm::LLVMConstPointerCast(v, glue_fn_ty.to_ref())
                }
              }
            };

        let tydesc = C_named_struct(ccx.tydesc_type(),
                                    [ti.size, // size
                                     ti.align, // align
                                     drop_glue, // drop_glue
                                     visit_glue, // visit_glue
                                     ti.name]); // name

        unsafe {
            let gvar = ti.tydesc;
            llvm::LLVMSetInitializer(gvar, tydesc);
            llvm::LLVMSetGlobalConstant(gvar, True);
            lib::llvm::SetLinkage(gvar, lib::llvm::InternalLinkage);
        }
    };
}
