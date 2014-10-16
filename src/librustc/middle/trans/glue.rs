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
use llvm::{ValueRef, True, get_param};
use llvm;
use middle::lang_items::ExchangeFreeFnLangItem;
use middle::subst;
use middle::subst::Subst;
use middle::trans::adt;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::cleanup;
use middle::trans::cleanup::CleanupMethods;
use middle::trans::common::*;
use middle::trans::datum;
use middle::trans::debuginfo;
use middle::trans::expr;
use middle::trans::machine::*;
use middle::trans::tvec;
use middle::trans::type_::Type;
use middle::trans::type_of::{type_of, sizing_type_of, align_of};
use middle::ty;
use util::ppaux::{ty_to_short_str, Repr};
use util::ppaux;

use arena::TypedArena;
use std::c_str::ToCStr;
use libc::c_uint;
use syntax::ast;
use syntax::parse::token;

pub fn trans_exchange_free_dyn<'blk, 'tcx>(cx: Block<'blk, 'tcx>, v: ValueRef,
                                           size: ValueRef, align: ValueRef)
                                           -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_exchange_free");
    let ccx = cx.ccx();
    callee::trans_lang_call(cx,
        langcall(cx, None, "", ExchangeFreeFnLangItem),
        [PointerCast(cx, v, Type::i8p(ccx)), size, align],
        Some(expr::Ignore)).bcx
}

pub fn trans_exchange_free<'blk, 'tcx>(cx: Block<'blk, 'tcx>, v: ValueRef,
                                       size: u64, align: u64) -> Block<'blk, 'tcx> {
    trans_exchange_free_dyn(cx, v, C_uint(cx.ccx(), size as uint),
                            C_uint(cx.ccx(), align as uint))
}

pub fn trans_exchange_free_ty<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, ptr: ValueRef,
                                          content_ty: ty::t) -> Block<'blk, 'tcx> {
    assert!(ty::type_is_sized(bcx.ccx().tcx(), content_ty));
    let sizing_type = sizing_type_of(bcx.ccx(), content_ty);
    let content_size = llsize_of_alloc(bcx.ccx(), sizing_type);

    // `Box<ZeroSizeType>` does not allocate.
    if content_size != 0 {
        let content_align = align_of(bcx.ccx(), content_ty);
        trans_exchange_free(bcx, ptr, content_size, content_align)
    } else {
        bcx
    }
}

pub fn get_drop_glue_type(ccx: &CrateContext, t: ty::t) -> ty::t {
    let tcx = ccx.tcx();
    // Even if there is no dtor for t, there might be one deeper down and we
    // might need to pass in the vtable ptr.
    if !ty::type_is_sized(tcx, t) {
        return t
    }
    if !ty::type_needs_drop(tcx, t) {
        return ty::mk_i8();
    }
    match ty::get(t).sty {
        ty::ty_uniq(typ) if !ty::type_needs_drop(tcx, typ)
                         && ty::type_is_sized(tcx, typ) => {
            let llty = sizing_type_of(ccx, typ);
            // `Box<ZeroSizeType>` does not allocate.
            if llsize_of_alloc(ccx, llty) == 0 {
                ty::mk_i8()
            } else {
                t
            }
        }
        _ => t
    }
}

pub fn drop_ty<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                           v: ValueRef,
                           t: ty::t,
                           source_location: Option<NodeInfo>)
                           -> Block<'blk, 'tcx> {
    // NB: v is an *alias* of type t here, not a direct value.
    debug!("drop_ty(t={})", t.repr(bcx.tcx()));
    let _icx = push_ctxt("drop_ty");
    if ty::type_needs_drop(bcx.tcx(), t) {
        let ccx = bcx.ccx();
        let glue = get_drop_glue(ccx, t);
        let glue_type = get_drop_glue_type(ccx, t);
        let ptr = if glue_type != t {
            PointerCast(bcx, v, type_of(ccx, glue_type).ptr_to())
        } else {
            v
        };

        match source_location {
            Some(sl) => debuginfo::set_source_location(bcx.fcx, sl.id, sl.span),
            None => debuginfo::clear_source_location(bcx.fcx)
        };

        Call(bcx, glue, [ptr], None);
    }
    bcx
}

pub fn drop_ty_immediate<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                     v: ValueRef,
                                     t: ty::t,
                                     source_location: Option<NodeInfo>)
                                     -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("drop_ty_immediate");
    let vp = alloca(bcx, type_of(bcx.ccx(), t), "");
    Store(bcx, v, vp);
    drop_ty(bcx, vp, t, source_location)
}

pub fn get_drop_glue(ccx: &CrateContext, t: ty::t) -> ValueRef {
    debug!("make drop glue for {}", ppaux::ty_to_string(ccx.tcx(), t));
    let t = get_drop_glue_type(ccx, t);
    debug!("drop glue type {}", ppaux::ty_to_string(ccx.tcx(), t));
    match ccx.drop_glues().borrow().find(&t) {
        Some(&glue) => return glue,
        _ => { }
    }

    let llty = if ty::type_is_sized(ccx.tcx(), t) {
        type_of(ccx, t).ptr_to()
    } else {
        type_of(ccx, ty::mk_uniq(ccx.tcx(), t)).ptr_to()
    };

    let llfnty = Type::glue_fn(ccx, llty);

    let (glue, new_sym) = match ccx.available_drop_glues().borrow().find(&t) {
        Some(old_sym) => {
            let glue = decl_cdecl_fn(ccx, old_sym.as_slice(), llfnty, ty::mk_nil());
            (glue, None)
        },
        None => {
            let (sym, glue) = declare_generic_glue(ccx, t, llfnty, "drop");
            (glue, Some(sym))
        },
    };

    ccx.drop_glues().borrow_mut().insert(t, glue);

    // To avoid infinite recursion, don't `make_drop_glue` until after we've
    // added the entry to the `drop_glues` cache.
    match new_sym {
        Some(sym) => {
            ccx.available_drop_glues().borrow_mut().insert(t, sym);
            // We're creating a new drop glue, so also generate a body.
            make_generic_glue(ccx, t, glue, make_drop_glue, "drop");
        },
        None => {},
    }

    glue
}

fn trans_struct_drop_flag<'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                      t: ty::t,
                                      v0: ValueRef,
                                      dtor_did: ast::DefId,
                                      class_did: ast::DefId,
                                      substs: &subst::Substs)
                                      -> Block<'blk, 'tcx> {
    let repr = adt::represent_type(bcx.ccx(), t);
    let struct_data = if ty::type_is_sized(bcx.tcx(), t) {
        v0
    } else {
        let llval = GEPi(bcx, v0, [0, abi::slice_elt_base]);
        Load(bcx, llval)
    };
    let drop_flag = unpack_datum!(bcx, adt::trans_drop_flag_ptr(bcx, &*repr, struct_data));
    with_cond(bcx, load_ty(bcx, drop_flag.val, ty::mk_bool()), |cx| {
        trans_struct_drop(cx, t, v0, dtor_did, class_did, substs)
    })
}

fn trans_struct_drop<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 t: ty::t,
                                 v0: ValueRef,
                                 dtor_did: ast::DefId,
                                 class_did: ast::DefId,
                                 substs: &subst::Substs)
                                 -> Block<'blk, 'tcx> {
    let repr = adt::represent_type(bcx.ccx(), t);

    // Find and call the actual destructor
    let dtor_addr = get_res_dtor(bcx.ccx(), dtor_did, t,
                                 class_did, substs);

    // The first argument is the "self" argument for drop
    let params = unsafe {
        let ty = Type::from_ref(llvm::LLVMTypeOf(dtor_addr));
        ty.element_type().func_params()
    };

    let fty = ty::lookup_item_type(bcx.tcx(), dtor_did).ty.subst(bcx.tcx(), substs);
    let self_ty = match ty::get(fty).sty {
        ty::ty_bare_fn(ref f) => {
            assert!(f.sig.inputs.len() == 1);
            f.sig.inputs[0]
        }
        _ => bcx.sess().bug(format!("Expected function type, found {}",
                                    bcx.ty_to_string(fty)).as_slice())
    };

    let (struct_data, info) = if ty::type_is_sized(bcx.tcx(), t) {
        (v0, None)
    } else {
        let data = GEPi(bcx, v0, [0, abi::slice_elt_base]);
        let info = GEPi(bcx, v0, [0, abi::slice_elt_len]);
        (Load(bcx, data), Some(Load(bcx, info)))
    };

    adt::fold_variants(bcx, &*repr, struct_data, |variant_cx, st, value| {
        // Be sure to put all of the fields into a scope so we can use an invoke
        // instruction to call the user destructor but still call the field
        // destructors if the user destructor fails.
        let field_scope = variant_cx.fcx.push_custom_cleanup_scope();

        // Class dtors have no explicit args, so the params should
        // just consist of the environment (self).
        assert_eq!(params.len(), 1);
        let self_arg = if ty::type_is_fat_ptr(bcx.tcx(), self_ty) {
            // The dtor expects a fat pointer, so make one, even if we have to fake it.
            let boxed_ty = ty::mk_open(bcx.tcx(), t);
            let scratch = datum::rvalue_scratch_datum(bcx, boxed_ty, "__fat_ptr_drop_self");
            Store(bcx, value, GEPi(bcx, scratch.val, [0, abi::slice_elt_base]));
            Store(bcx,
                  // If we just had a thin pointer, make a fat pointer by sticking
                  // null where we put the unsizing info. This works because t
                  // is a sized type, so we will only unpack the fat pointer, never
                  // use the fake info.
                  info.unwrap_or(C_null(Type::i8p(bcx.ccx()))),
                  GEPi(bcx, scratch.val, [0, abi::slice_elt_len]));
            PointerCast(variant_cx, scratch.val, *params.get(0))
        } else {
            PointerCast(variant_cx, value, *params.get(0))
        };
        let args = vec!(self_arg);

        // Add all the fields as a value which needs to be cleaned at the end of
        // this scope. Iterate in reverse order so a Drop impl doesn't reverse
        // the order in which fields get dropped.
        for (i, ty) in st.fields.iter().enumerate().rev() {
            let llfld_a = adt::struct_field_ptr(variant_cx, &*st, value, i, false);

            let val = if ty::type_is_sized(bcx.tcx(), *ty) {
                llfld_a
            } else {
                let boxed_ty = ty::mk_open(bcx.tcx(), *ty);
                let scratch = datum::rvalue_scratch_datum(bcx, boxed_ty, "__fat_ptr_drop_field");
                Store(bcx, llfld_a, GEPi(bcx, scratch.val, [0, abi::slice_elt_base]));
                Store(bcx, info.unwrap(), GEPi(bcx, scratch.val, [0, abi::slice_elt_len]));
                scratch.val
            };
            variant_cx.fcx.schedule_drop_mem(cleanup::CustomScope(field_scope),
                                             val, *ty);
        }

        let dtor_ty = ty::mk_ctor_fn(variant_cx.tcx(), ast::DUMMY_NODE_ID,
                                     [get_drop_glue_type(bcx.ccx(), t)], ty::mk_nil());
        let (_, variant_cx) = invoke(variant_cx, dtor_addr, args, dtor_ty, None, false);

        variant_cx.fcx.pop_and_trans_custom_cleanup_scope(variant_cx, field_scope);
        variant_cx
    })
}

fn size_and_align_of_dst(bcx: Block, t :ty::t, info: ValueRef) -> (ValueRef, ValueRef) {
    debug!("calculate size of DST: {}; with lost info: {}",
           bcx.ty_to_string(t), bcx.val_to_string(info));
    if ty::type_is_sized(bcx.tcx(), t) {
        let sizing_type = sizing_type_of(bcx.ccx(), t);
        let size = C_uint(bcx.ccx(), llsize_of_alloc(bcx.ccx(), sizing_type) as uint);
        let align = C_uint(bcx.ccx(), align_of(bcx.ccx(), t) as uint);
        return (size, align);
    }
    match ty::get(t).sty {
        ty::ty_struct(id, ref substs) => {
            let ccx = bcx.ccx();
            // First get the size of all statically known fields.
            // Don't use type_of::sizing_type_of because that expects t to be sized.
            assert!(!ty::type_is_simd(bcx.tcx(), t));
            let repr = adt::represent_type(ccx, t);
            let sizing_type = adt::sizing_type_of(ccx, &*repr, true);
            let sized_size = C_uint(ccx, llsize_of_alloc(ccx, sizing_type) as uint);
            let sized_align = C_uint(ccx, llalign_of_min(ccx, sizing_type) as uint);

            // Recurse to get the size of the dynamically sized field (must be
            // the last field).
            let fields = ty::struct_fields(bcx.tcx(), id, substs);
            let last_field = fields[fields.len()-1];
            let field_ty = last_field.mt.ty;
            let (unsized_size, unsized_align) = size_and_align_of_dst(bcx, field_ty, info);

            // Return the sum of sizes and max of aligns.
            let size = Add(bcx, sized_size, unsized_size);
            let align = Select(bcx,
                               ICmp(bcx, llvm::IntULT, sized_align, unsized_align),
                               sized_align,
                               unsized_align);
            (size, align)
        }
        ty::ty_trait(..) => {
            // info points to the vtable and the second entry in the vtable is the
            // dynamic size of the object.
            let info = PointerCast(bcx, info, Type::int(bcx.ccx()).ptr_to());
            let size_ptr = GEPi(bcx, info, [1u]);
            let align_ptr = GEPi(bcx, info, [2u]);
            (Load(bcx, size_ptr), Load(bcx, align_ptr))
        }
        ty::ty_vec(unit_ty, None) => {
            // The info in this case is the length of the vec, so the size is that
            // times the unit size.
            let llunit_ty = sizing_type_of(bcx.ccx(), unit_ty);
            let unit_size = llsize_of_alloc(bcx.ccx(), llunit_ty);
            (Mul(bcx, info, C_uint(bcx.ccx(), unit_size as uint)), C_uint(bcx.ccx(), 8))
        }
        _ => bcx.sess().bug(format!("Unexpected unsized type, found {}",
                                    bcx.ty_to_string(t)).as_slice())
    }
}

fn make_drop_glue<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, v0: ValueRef, t: ty::t)
                              -> Block<'blk, 'tcx> {
    // NB: v0 is an *alias* of type t here, not a direct value.
    let _icx = push_ctxt("make_drop_glue");
    match ty::get(t).sty {
        ty::ty_uniq(content_ty) => {
            match ty::get(content_ty).sty {
                ty::ty_vec(ty, None) => {
                    tvec::make_drop_glue_unboxed(bcx, v0, ty, true)
                }
                ty::ty_str => {
                    let unit_ty = ty::sequence_element_type(bcx.tcx(), content_ty);
                    tvec::make_drop_glue_unboxed(bcx, v0, unit_ty, true)
                }
                ty::ty_trait(..) => {
                    let lluniquevalue = GEPi(bcx, v0, [0, abi::trt_field_box]);
                    // Only drop the value when it is non-null
                    let concrete_ptr = Load(bcx, lluniquevalue);
                    with_cond(bcx, IsNotNull(bcx, concrete_ptr), |bcx| {
                        let dtor_ptr = Load(bcx, GEPi(bcx, v0, [0, abi::trt_field_vtable]));
                        let dtor = Load(bcx, dtor_ptr);
                        Call(bcx,
                             dtor,
                             [PointerCast(bcx, lluniquevalue, Type::i8p(bcx.ccx()))],
                             None);
                        bcx
                    })
                }
                ty::ty_struct(..) if !ty::type_is_sized(bcx.tcx(), content_ty) => {
                    let llval = GEPi(bcx, v0, [0, abi::slice_elt_base]);
                    let llbox = Load(bcx, llval);
                    let not_null = IsNotNull(bcx, llbox);
                    with_cond(bcx, not_null, |bcx| {
                        let bcx = drop_ty(bcx, v0, content_ty, None);
                        let info = GEPi(bcx, v0, [0, abi::slice_elt_len]);
                        let info = Load(bcx, info);
                        let (llsize, llalign) = size_and_align_of_dst(bcx, content_ty, info);
                        trans_exchange_free_dyn(bcx, llbox, llsize, llalign)
                    })
                }
                _ => {
                    assert!(ty::type_is_sized(bcx.tcx(), content_ty));
                    let llval = v0;
                    let llbox = Load(bcx, llval);
                    let not_null = IsNotNull(bcx, llbox);
                    with_cond(bcx, not_null, |bcx| {
                        let bcx = drop_ty(bcx, llbox, content_ty, None);
                        trans_exchange_free_ty(bcx, llbox, content_ty)
                    })
                }
            }
        }
        ty::ty_struct(did, ref substs) | ty::ty_enum(did, ref substs) => {
            let tcx = bcx.tcx();
            match ty::ty_dtor(tcx, did) {
                ty::TraitDtor(dtor, true) => {
                    // FIXME(16758) Since the struct is unsized, it is hard to
                    // find the drop flag (which is at the end of the struct).
                    // Lets just ignore the flag and pretend everything will be
                    // OK.
                    if ty::type_is_sized(bcx.tcx(), t) {
                        trans_struct_drop_flag(bcx, t, v0, dtor, did, substs)
                    } else {
                        // Give the user a heads up that we are doing something
                        // stupid and dangerous.
                        bcx.sess().warn(format!("Ignoring drop flag in destructor for {}\
                                                 because the struct is unsized. See issue\
                                                 #16758",
                                                bcx.ty_to_string(t)).as_slice());
                        trans_struct_drop(bcx, t, v0, dtor, did, substs)
                    }
                }
                ty::TraitDtor(dtor, false) => {
                    trans_struct_drop(bcx, t, v0, dtor, did, substs)
                }
                ty::NoDtor => {
                    // No dtor? Just the default case
                    iter_structural_ty(bcx, v0, t, |bb, vv, tt| drop_ty(bb, vv, tt, None))
                }
            }
        }
        ty::ty_unboxed_closure(..) => iter_structural_ty(bcx,
                                                         v0,
                                                         t,
                                                         |bb, vv, tt| drop_ty(bb, vv, tt, None)),
        ty::ty_closure(ref f) if f.store == ty::UniqTraitStore => {
            let box_cell_v = GEPi(bcx, v0, [0u, abi::fn_field_box]);
            let env = Load(bcx, box_cell_v);
            let env_ptr_ty = Type::at_box(bcx.ccx(), Type::i8(bcx.ccx())).ptr_to();
            let env = PointerCast(bcx, env, env_ptr_ty);
            with_cond(bcx, IsNotNull(bcx, env), |bcx| {
                let dtor_ptr = GEPi(bcx, env, [0u, abi::box_field_drop_glue]);
                let dtor = Load(bcx, dtor_ptr);
                Call(bcx, dtor, [PointerCast(bcx, box_cell_v, Type::i8p(bcx.ccx()))], None);
                bcx
            })
        }
        ty::ty_trait(..) => {
            // No need to do a null check here (as opposed to the Box<trait case
            // above), because this happens for a trait field in an unsized
            // struct. If anything is null, it is the whole struct and we won't
            // get here.
            let lluniquevalue = GEPi(bcx, v0, [0, abi::trt_field_box]);
            let dtor_ptr = Load(bcx, GEPi(bcx, v0, [0, abi::trt_field_vtable]));
            let dtor = Load(bcx, dtor_ptr);
            Call(bcx,
                 dtor,
                 [PointerCast(bcx, Load(bcx, lluniquevalue), Type::i8p(bcx.ccx()))],
                 None);
            bcx
        }
        ty::ty_vec(ty, None) => tvec::make_drop_glue_unboxed(bcx, v0, ty, false),
        _ => {
            assert!(ty::type_is_sized(bcx.tcx(), t));
            if ty::type_needs_drop(bcx.tcx(), t) &&
                ty::type_is_structural(t) {
                iter_structural_ty(bcx, v0, t, |bb, vv, tt| drop_ty(bb, vv, tt, None))
            } else {
                bcx
            }
        }
    }
}

// Generates the declaration for (but doesn't emit) a type descriptor.
pub fn declare_tydesc(ccx: &CrateContext, t: ty::t) -> tydesc_info {
    // If emit_tydescs already ran, then we shouldn't be creating any new
    // tydescs.
    assert!(!ccx.finished_tydescs().get());

    let llty = type_of(ccx, t);

    if ccx.sess().count_type_sizes() {
        println!("{}\t{}", llsize_of_real(ccx, llty),
                 ppaux::ty_to_string(ccx.tcx(), t));
    }

    let llsize = llsize_of(ccx, llty);
    let llalign = llalign_of(ccx, llty);
    let name = mangle_internal_name_by_type_and_seq(ccx, t, "tydesc");
    debug!("+++ declare_tydesc {} {}", ppaux::ty_to_string(ccx.tcx(), t), name);
    let gvar = name.as_slice().with_c_str(|buf| {
        unsafe {
            llvm::LLVMAddGlobal(ccx.llmod(), ccx.tydesc_type().to_ref(), buf)
        }
    });
    note_unique_llvm_symbol(ccx, name);

    let ty_name = token::intern_and_get_ident(
        ppaux::ty_to_string(ccx.tcx(), t).as_slice());
    let ty_name = C_str_slice(ccx, ty_name);

    debug!("--- declare_tydesc {}", ppaux::ty_to_string(ccx.tcx(), t));
    tydesc_info {
        ty: t,
        tydesc: gvar,
        size: llsize,
        align: llalign,
        name: ty_name,
    }
}

fn declare_generic_glue(ccx: &CrateContext, t: ty::t, llfnty: Type,
                        name: &str) -> (String, ValueRef) {
    let _icx = push_ctxt("declare_generic_glue");
    let fn_nm = mangle_internal_name_by_type_and_seq(
        ccx,
        t,
        format!("glue_{}", name).as_slice());
    let llfn = decl_cdecl_fn(ccx, fn_nm.as_slice(), llfnty, ty::mk_nil());
    note_unique_llvm_symbol(ccx, fn_nm.clone());
    return (fn_nm, llfn);
}

fn make_generic_glue(ccx: &CrateContext,
                     t: ty::t,
                     llfn: ValueRef,
                     helper: <'blk, 'tcx> |Block<'blk, 'tcx>, ValueRef, ty::t|
                                           -> Block<'blk, 'tcx>,
                     name: &str)
                     -> ValueRef {
    let _icx = push_ctxt("make_generic_glue");
    let glue_name = format!("glue {} {}", name, ty_to_short_str(ccx.tcx(), t));
    let _s = StatRecorder::new(ccx, glue_name);

    let arena = TypedArena::new();
    let empty_param_substs = param_substs::empty();
    let fcx = new_fn_ctxt(ccx, llfn, ast::DUMMY_NODE_ID, false, ty::mk_nil(),
                          &empty_param_substs, None, &arena);

    let bcx = init_function(&fcx, false, ty::mk_nil());

    update_linkage(ccx, llfn, None, OriginalTranslation);

    ccx.stats().n_glues_created.set(ccx.stats().n_glues_created.get() + 1u);
    // All glue functions take values passed *by alias*; this is a
    // requirement since in many contexts glue is invoked indirectly and
    // the caller has no idea if it's dealing with something that can be
    // passed by value.
    //
    // llfn is expected be declared to take a parameter of the appropriate
    // type, so we don't need to explicitly cast the function parameter.

    let llrawptr0 = get_param(llfn, fcx.arg_pos(0) as c_uint);
    let bcx = helper(bcx, llrawptr0, t);
    finish_fn(&fcx, bcx, ty::mk_nil());

    llfn
}

pub fn emit_tydescs(ccx: &CrateContext) {
    let _icx = push_ctxt("emit_tydescs");
    // As of this point, allow no more tydescs to be created.
    ccx.finished_tydescs().set(true);
    let glue_fn_ty = Type::generic_glue_fn(ccx).ptr_to();
    for (_, ti) in ccx.tydescs().borrow().iter() {
        // Each of the glue functions needs to be cast to a generic type
        // before being put into the tydesc because we only have a singleton
        // tydesc type. Then we'll recast each function to its real type when
        // calling it.
        let drop_glue = unsafe {
            llvm::LLVMConstPointerCast(get_drop_glue(ccx, ti.ty), glue_fn_ty.to_ref())
        };
        ccx.stats().n_real_glues.set(ccx.stats().n_real_glues.get() + 1);

        let tydesc = C_named_struct(ccx.tydesc_type(),
                                    [ti.size, // size
                                     ti.align, // align
                                     drop_glue, // drop_glue
                                     ti.name]); // name

        unsafe {
            let gvar = ti.tydesc;
            llvm::LLVMSetInitializer(gvar, tydesc);
            llvm::LLVMSetGlobalConstant(gvar, True);
            llvm::SetLinkage(gvar, llvm::InternalLinkage);
        }
    };
}
