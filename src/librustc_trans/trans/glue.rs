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
// Code relating to drop glue.


use back::abi;
use back::link::*;
use llvm;
use llvm::{ValueRef, get_param};
use metadata::csearch;
use middle::lang_items::ExchangeFreeFnLangItem;
use middle::subst;
use middle::subst::{Subst, Substs};
use middle::ty::{self, Ty};
use trans::adt;
use trans::adt::GetDtorType; // for tcx.dtor_type()
use trans::base::*;
use trans::build::*;
use trans::callee;
use trans::cleanup;
use trans::cleanup::CleanupMethods;
use trans::common::*;
use trans::debuginfo::DebugLoc;
use trans::declare;
use trans::expr;
use trans::foreign;
use trans::inline;
use trans::machine::*;
use trans::monomorphize;
use trans::type_of::{type_of, type_of_dtor, sizing_type_of, align_of};
use trans::type_::Type;
use util::ppaux;
use util::ppaux::{ty_to_short_str, Repr};

use arena::TypedArena;
use libc::c_uint;
use syntax::ast;

pub fn trans_exchange_free_dyn<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                                           v: ValueRef,
                                           size: ValueRef,
                                           align: ValueRef,
                                           debug_loc: DebugLoc)
                                           -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_exchange_free");
    let ccx = cx.ccx();
    callee::trans_lang_call(cx,
        langcall(cx, None, "", ExchangeFreeFnLangItem),
        &[PointerCast(cx, v, Type::i8p(ccx)), size, align],
        Some(expr::Ignore),
        debug_loc).bcx
}

pub fn trans_exchange_free<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                                       v: ValueRef,
                                       size: u64,
                                       align: u32,
                                       debug_loc: DebugLoc)
                                       -> Block<'blk, 'tcx> {
    trans_exchange_free_dyn(cx,
                            v,
                            C_uint(cx.ccx(), size),
                            C_uint(cx.ccx(), align),
                            debug_loc)
}

pub fn trans_exchange_free_ty<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                          ptr: ValueRef,
                                          content_ty: Ty<'tcx>,
                                          debug_loc: DebugLoc)
                                          -> Block<'blk, 'tcx> {
    assert!(type_is_sized(bcx.ccx().tcx(), content_ty));
    let sizing_type = sizing_type_of(bcx.ccx(), content_ty);
    let content_size = llsize_of_alloc(bcx.ccx(), sizing_type);

    // `Box<ZeroSizeType>` does not allocate.
    if content_size != 0 {
        let content_align = align_of(bcx.ccx(), content_ty);
        trans_exchange_free(bcx, ptr, content_size, content_align, debug_loc)
    } else {
        bcx
    }
}

pub fn get_drop_glue_type<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                    t: Ty<'tcx>) -> Ty<'tcx> {
    let tcx = ccx.tcx();
    // Even if there is no dtor for t, there might be one deeper down and we
    // might need to pass in the vtable ptr.
    if !type_is_sized(tcx, t) {
        return t
    }

    // FIXME (#22815): note that type_needs_drop conservatively
    // approximates in some cases and may say a type expression
    // requires drop glue when it actually does not.
    //
    // (In this case it is not clear whether any harm is done, i.e.
    // erroneously returning `t` in some cases where we could have
    // returned `tcx.types.i8` does not appear unsound. The impact on
    // code quality is unknown at this time.)

    if !type_needs_drop(tcx, t) {
        return tcx.types.i8;
    }
    match t.sty {
        ty::ty_uniq(typ) if !type_needs_drop(tcx, typ)
                         && type_is_sized(tcx, typ) => {
            let llty = sizing_type_of(ccx, typ);
            // `Box<ZeroSizeType>` does not allocate.
            if llsize_of_alloc(ccx, llty) == 0 {
                tcx.types.i8
            } else {
                t
            }
        }
        _ => t
    }
}

pub fn drop_ty<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                           v: ValueRef,
                           t: Ty<'tcx>,
                           debug_loc: DebugLoc) -> Block<'blk, 'tcx> {
    drop_ty_core(bcx, v, t, debug_loc, false)
}

pub fn drop_ty_core<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                v: ValueRef,
                                t: Ty<'tcx>,
                                debug_loc: DebugLoc,
                                skip_dtor: bool) -> Block<'blk, 'tcx> {
    // NB: v is an *alias* of type t here, not a direct value.
    debug!("drop_ty_core(t={}, skip_dtor={})", t.repr(bcx.tcx()), skip_dtor);
    let _icx = push_ctxt("drop_ty");
    if bcx.fcx.type_needs_drop(t) {
        let ccx = bcx.ccx();
        let g = if skip_dtor {
            DropGlueKind::TyContents(t)
        } else {
            DropGlueKind::Ty(t)
        };
        let glue = get_drop_glue_core(ccx, g);
        let glue_type = get_drop_glue_type(ccx, t);
        let ptr = if glue_type != t {
            PointerCast(bcx, v, type_of(ccx, glue_type).ptr_to())
        } else {
            v
        };

        Call(bcx, glue, &[ptr], None, debug_loc);
    }
    bcx
}

pub fn drop_ty_immediate<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                     v: ValueRef,
                                     t: Ty<'tcx>,
                                     debug_loc: DebugLoc,
                                     skip_dtor: bool)
                                     -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("drop_ty_immediate");
    let vp = alloca(bcx, type_of(bcx.ccx(), t), "");
    store_ty(bcx, v, vp, t);
    drop_ty_core(bcx, vp, t, debug_loc, skip_dtor)
}

pub fn get_drop_glue<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>) -> ValueRef {
    get_drop_glue_core(ccx, DropGlueKind::Ty(t))
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum DropGlueKind<'tcx> {
    /// The normal path; runs the dtor, and then recurs on the contents
    Ty(Ty<'tcx>),
    /// Skips the dtor, if any, for ty; drops the contents directly.
    /// Note that the dtor is only skipped at the most *shallow*
    /// level, namely, an `impl Drop for Ty` itself. So, for example,
    /// if Ty is Newtype(S) then only the Drop impl for for Newtype
    /// itself will be skipped, while the Drop impl for S, if any,
    /// will be invoked.
    TyContents(Ty<'tcx>),
}

impl<'tcx> DropGlueKind<'tcx> {
    fn ty(&self) -> Ty<'tcx> {
        match *self { DropGlueKind::Ty(t) | DropGlueKind::TyContents(t) => t }
    }

    fn map_ty<F>(&self, mut f: F) -> DropGlueKind<'tcx> where F: FnMut(Ty<'tcx>) -> Ty<'tcx>
    {
        match *self {
            DropGlueKind::Ty(t) => DropGlueKind::Ty(f(t)),
            DropGlueKind::TyContents(t) => DropGlueKind::TyContents(f(t)),
        }
    }

    fn to_string<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> String {
        let t_str = ppaux::ty_to_string(ccx.tcx(), self.ty());
        match *self {
            DropGlueKind::Ty(_) => format!("DropGlueKind::Ty({})", t_str),
            DropGlueKind::TyContents(_) => format!("DropGlueKind::TyContents({})", t_str),
        }
    }
}

fn get_drop_glue_core<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                g: DropGlueKind<'tcx>) -> ValueRef {
    debug!("make drop glue for {}", g.to_string(ccx));
    let g = g.map_ty(|t| get_drop_glue_type(ccx, t));
    debug!("drop glue type {}", g.to_string(ccx));
    match ccx.drop_glues().borrow().get(&g) {
        Some(&glue) => return glue,
        _ => { }
    }
    let t = g.ty();

    let llty = if type_is_sized(ccx.tcx(), t) {
        type_of(ccx, t).ptr_to()
    } else {
        type_of(ccx, ty::mk_uniq(ccx.tcx(), t)).ptr_to()
    };

    let llfnty = Type::glue_fn(ccx, llty);

    // To avoid infinite recursion, don't `make_drop_glue` until after we've
    // added the entry to the `drop_glues` cache.
    if let Some(old_sym) = ccx.available_drop_glues().borrow().get(&g) {
        let llfn = declare::declare_cfn(ccx, &old_sym, llfnty, ty::mk_nil(ccx.tcx()));
        ccx.drop_glues().borrow_mut().insert(g, llfn);
        return llfn;
    };

    let fn_nm = mangle_internal_name_by_type_and_seq(ccx, t, "drop");
    let llfn = declare::define_cfn(ccx, &fn_nm, llfnty, ty::mk_nil(ccx.tcx())).unwrap_or_else(||{
       ccx.sess().bug(&format!("symbol `{}` already defined", fn_nm));
    });
    ccx.available_drop_glues().borrow_mut().insert(g, fn_nm);

    let _s = StatRecorder::new(ccx, format!("drop {}", ty_to_short_str(ccx.tcx(), t)));

    let empty_substs = ccx.tcx().mk_substs(Substs::trans_empty());
    let (arena, fcx): (TypedArena<_>, FunctionContext);
    arena = TypedArena::new();
    fcx = new_fn_ctxt(ccx, llfn, ast::DUMMY_NODE_ID, false,
                      ty::FnConverging(ty::mk_nil(ccx.tcx())),
                      empty_substs, None, &arena);

    let bcx = init_function(&fcx, false, ty::FnConverging(ty::mk_nil(ccx.tcx())));

    update_linkage(ccx, llfn, None, OriginalTranslation);

    ccx.stats().n_glues_created.set(ccx.stats().n_glues_created.get() + 1);
    // All glue functions take values passed *by alias*; this is a
    // requirement since in many contexts glue is invoked indirectly and
    // the caller has no idea if it's dealing with something that can be
    // passed by value.
    //
    // llfn is expected be declared to take a parameter of the appropriate
    // type, so we don't need to explicitly cast the function parameter.

    let llrawptr0 = get_param(llfn, fcx.arg_pos(0) as c_uint);
    let bcx = make_drop_glue(bcx, llrawptr0, g);
    finish_fn(&fcx, bcx, ty::FnConverging(ty::mk_nil(ccx.tcx())), DebugLoc::None);

    llfn
}

fn trans_struct_drop_flag<'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                      t: Ty<'tcx>,
                                      v0: ValueRef,
                                      dtor_did: ast::DefId,
                                      class_did: ast::DefId,
                                      substs: &subst::Substs<'tcx>)
                                      -> Block<'blk, 'tcx> {
    let repr = adt::represent_type(bcx.ccx(), t);
    let struct_data = if type_is_sized(bcx.tcx(), t) {
        v0
    } else {
        let llval = GEPi(bcx, v0, &[0, abi::FAT_PTR_ADDR]);
        Load(bcx, llval)
    };
    let drop_flag = unpack_datum!(bcx, adt::trans_drop_flag_ptr(bcx, &*repr, struct_data));
    let loaded = load_ty(bcx, drop_flag.val, bcx.tcx().dtor_type());
    let drop_flag_llty = type_of(bcx.fcx.ccx, bcx.tcx().dtor_type());
    let init_val = C_integral(drop_flag_llty, adt::DTOR_NEEDED as u64, false);

    let bcx = if !bcx.ccx().check_drop_flag_for_sanity() {
        bcx
    } else {
        let drop_flag_llty = type_of(bcx.fcx.ccx, bcx.tcx().dtor_type());
        let done_val = C_integral(drop_flag_llty, adt::DTOR_DONE as u64, false);
        let not_init = ICmp(bcx, llvm::IntNE, loaded, init_val, DebugLoc::None);
        let not_done = ICmp(bcx, llvm::IntNE, loaded, done_val, DebugLoc::None);
        let drop_flag_neither_initialized_nor_cleared =
            And(bcx, not_init, not_done, DebugLoc::None);
        with_cond(bcx, drop_flag_neither_initialized_nor_cleared, |cx| {
            let llfn = cx.ccx().get_intrinsic(&("llvm.debugtrap"));
            Call(cx, llfn, &[], None, DebugLoc::None);
            cx
        })
    };

    let drop_flag_dtor_needed = ICmp(bcx, llvm::IntEQ, loaded, init_val, DebugLoc::None);
    with_cond(bcx, drop_flag_dtor_needed, |cx| {
        trans_struct_drop(cx, t, v0, dtor_did, class_did, substs)
    })

}

pub fn get_res_dtor<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                              did: ast::DefId,
                              t: Ty<'tcx>,
                              parent_id: ast::DefId,
                              substs: &Substs<'tcx>)
                              -> ValueRef {
    let _icx = push_ctxt("trans_res_dtor");
    let did = inline::maybe_instantiate_inline(ccx, did);

    if !substs.types.is_empty() {
        assert_eq!(did.krate, ast::LOCAL_CRATE);

        // Since we're in trans we don't care for any region parameters
        let substs = ccx.tcx().mk_substs(Substs::erased(substs.types.clone()));

        let (val, _, _) = monomorphize::monomorphic_fn(ccx, did, substs, None);

        val
    } else if did.krate == ast::LOCAL_CRATE {
        get_item_val(ccx, did.node)
    } else {
        let tcx = ccx.tcx();
        let name = csearch::get_symbol(&ccx.sess().cstore, did);
        let class_ty = ty::lookup_item_type(tcx, parent_id).ty.subst(tcx, substs);
        let llty = type_of_dtor(ccx, class_ty);
        let dtor_ty = ty::mk_ctor_fn(ccx.tcx(),
                                     did,
                                     &[get_drop_glue_type(ccx, t)],
                                     ty::mk_nil(ccx.tcx()));
        foreign::get_extern_fn(ccx, &mut *ccx.externs().borrow_mut(), &name[..], llvm::CCallConv,
                               llty, dtor_ty)
    }
}

fn trans_struct_drop<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 t: Ty<'tcx>,
                                 v0: ValueRef,
                                 dtor_did: ast::DefId,
                                 class_did: ast::DefId,
                                 substs: &subst::Substs<'tcx>)
                                 -> Block<'blk, 'tcx>
{
    debug!("trans_struct_drop t: {}", bcx.ty_to_string(t));

    // Find and call the actual destructor
    let dtor_addr = get_res_dtor(bcx.ccx(), dtor_did, t, class_did, substs);

    // Class dtors have no explicit args, so the params should
    // just consist of the environment (self).
    let params = unsafe {
        let ty = Type::from_ref(llvm::LLVMTypeOf(dtor_addr));
        ty.element_type().func_params()
    };
    assert_eq!(params.len(), 1);

    // Be sure to put the contents into a scope so we can use an invoke
    // instruction to call the user destructor but still call the field
    // destructors if the user destructor panics.
    //
    // FIXME (#14875) panic-in-drop semantics might be unsupported; we
    // might well consider changing below to more direct code.
    let contents_scope = bcx.fcx.push_custom_cleanup_scope();

    // Issue #23611: schedule cleanup of contents, re-inspecting the
    // discriminant (if any) in case of variant swap in drop code.
    bcx.fcx.schedule_drop_enum_contents(cleanup::CustomScope(contents_scope), v0, t);

    let glue_type = get_drop_glue_type(bcx.ccx(), t);
    let dtor_ty = ty::mk_ctor_fn(bcx.tcx(), class_did, &[glue_type], ty::mk_nil(bcx.tcx()));
    let (_, bcx) = invoke(bcx, dtor_addr, &[v0], dtor_ty, DebugLoc::None);

    bcx.fcx.pop_and_trans_custom_cleanup_scope(bcx, contents_scope)
}

fn size_and_align_of_dst<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, t: Ty<'tcx>, info: ValueRef)
                                     -> (ValueRef, ValueRef) {
    debug!("calculate size of DST: {}; with lost info: {}",
           bcx.ty_to_string(t), bcx.val_to_string(info));
    if type_is_sized(bcx.tcx(), t) {
        let sizing_type = sizing_type_of(bcx.ccx(), t);
        let size = C_uint(bcx.ccx(), llsize_of_alloc(bcx.ccx(), sizing_type));
        let align = C_uint(bcx.ccx(), align_of(bcx.ccx(), t));
        return (size, align);
    }
    match t.sty {
        ty::ty_struct(id, substs) => {
            let ccx = bcx.ccx();
            // First get the size of all statically known fields.
            // Don't use type_of::sizing_type_of because that expects t to be sized.
            assert!(!ty::type_is_simd(bcx.tcx(), t));
            let repr = adt::represent_type(ccx, t);
            let sizing_type = adt::sizing_type_of(ccx, &*repr, true);
            let sized_size = C_uint(ccx, llsize_of_alloc(ccx, sizing_type));
            let sized_align = C_uint(ccx, llalign_of_min(ccx, sizing_type));

            // Recurse to get the size of the dynamically sized field (must be
            // the last field).
            let fields = ty::struct_fields(bcx.tcx(), id, substs);
            let last_field = fields[fields.len()-1];
            let field_ty = last_field.mt.ty;
            let (unsized_size, unsized_align) = size_and_align_of_dst(bcx, field_ty, info);

            // Return the sum of sizes and max of aligns.
            let size = Add(bcx, sized_size, unsized_size, DebugLoc::None);
            let align = Select(bcx,
                               ICmp(bcx,
                                    llvm::IntULT,
                                    sized_align,
                                    unsized_align,
                                    DebugLoc::None),
                               sized_align,
                               unsized_align);
            (size, align)
        }
        ty::ty_trait(..) => {
            // info points to the vtable and the second entry in the vtable is the
            // dynamic size of the object.
            let info = PointerCast(bcx, info, Type::int(bcx.ccx()).ptr_to());
            let size_ptr = GEPi(bcx, info, &[1]);
            let align_ptr = GEPi(bcx, info, &[2]);
            (Load(bcx, size_ptr), Load(bcx, align_ptr))
        }
        ty::ty_vec(_, None) | ty::ty_str => {
            let unit_ty = ty::sequence_element_type(bcx.tcx(), t);
            // The info in this case is the length of the str, so the size is that
            // times the unit size.
            let llunit_ty = sizing_type_of(bcx.ccx(), unit_ty);
            let unit_align = llalign_of_min(bcx.ccx(), llunit_ty);
            let unit_size = llsize_of_alloc(bcx.ccx(), llunit_ty);
            (Mul(bcx, info, C_uint(bcx.ccx(), unit_size), DebugLoc::None),
             C_uint(bcx.ccx(), unit_align))
        }
        _ => bcx.sess().bug(&format!("Unexpected unsized type, found {}",
                                    bcx.ty_to_string(t)))
    }
}

fn make_drop_glue<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, v0: ValueRef, g: DropGlueKind<'tcx>)
                              -> Block<'blk, 'tcx> {
    let t = g.ty();
    let skip_dtor = match g { DropGlueKind::Ty(_) => false, DropGlueKind::TyContents(_) => true };
    // NB: v0 is an *alias* of type t here, not a direct value.
    let _icx = push_ctxt("make_drop_glue");

    // Only drop the value when it ... well, we used to check for
    // non-null, (and maybe we need to continue doing so), but we now
    // must definitely check for special bit-patterns corresponding to
    // the special dtor markings.

    let inttype = Type::int(bcx.ccx());
    let dropped_pattern = C_integral(inttype, adt::dtor_done_usize(bcx.fcx.ccx) as u64, false);

    match t.sty {
        ty::ty_uniq(content_ty) => {
            // Support for ty_uniq is built-in and its drop glue is
            // special. It may move to library and have Drop impl. As
            // a safe-guard, assert ty_uniq not used with TyContents.
            assert!(!skip_dtor);
            if !type_is_sized(bcx.tcx(), content_ty) {
                let llval = GEPi(bcx, v0, &[0, abi::FAT_PTR_ADDR]);
                let llbox = Load(bcx, llval);
                let llbox_as_usize = PtrToInt(bcx, llbox, Type::int(bcx.ccx()));
                let drop_flag_not_dropped_already =
                    ICmp(bcx, llvm::IntNE, llbox_as_usize, dropped_pattern, DebugLoc::None);
                with_cond(bcx, drop_flag_not_dropped_already, |bcx| {
                    let bcx = drop_ty(bcx, v0, content_ty, DebugLoc::None);
                    let info = GEPi(bcx, v0, &[0, abi::FAT_PTR_EXTRA]);
                    let info = Load(bcx, info);
                    let (llsize, llalign) = size_and_align_of_dst(bcx, content_ty, info);

                    // `Box<ZeroSizeType>` does not allocate.
                    let needs_free = ICmp(bcx,
                                          llvm::IntNE,
                                          llsize,
                                          C_uint(bcx.ccx(), 0u64),
                                          DebugLoc::None);
                    with_cond(bcx, needs_free, |bcx| {
                        trans_exchange_free_dyn(bcx, llbox, llsize, llalign, DebugLoc::None)
                    })
                })
            } else {
                let llval = v0;
                let llbox = Load(bcx, llval);
                let llbox_as_usize = PtrToInt(bcx, llbox, inttype);
                let drop_flag_not_dropped_already =
                    ICmp(bcx, llvm::IntNE, llbox_as_usize, dropped_pattern, DebugLoc::None);
                with_cond(bcx, drop_flag_not_dropped_already, |bcx| {
                    let bcx = drop_ty(bcx, llbox, content_ty, DebugLoc::None);
                    trans_exchange_free_ty(bcx, llbox, content_ty, DebugLoc::None)
                })
            }
        }
        ty::ty_struct(did, substs) | ty::ty_enum(did, substs) => {
            let tcx = bcx.tcx();
            match (ty::ty_dtor(tcx, did), skip_dtor) {
                (ty::TraitDtor(dtor, true), false) => {
                    // FIXME(16758) Since the struct is unsized, it is hard to
                    // find the drop flag (which is at the end of the struct).
                    // Lets just ignore the flag and pretend everything will be
                    // OK.
                    if type_is_sized(bcx.tcx(), t) {
                        trans_struct_drop_flag(bcx, t, v0, dtor, did, substs)
                    } else {
                        // Give the user a heads up that we are doing something
                        // stupid and dangerous.
                        bcx.sess().warn(&format!("Ignoring drop flag in destructor for {}\
                                                 because the struct is unsized. See issue\
                                                 #16758",
                                                bcx.ty_to_string(t)));
                        trans_struct_drop(bcx, t, v0, dtor, did, substs)
                    }
                }
                (ty::TraitDtor(dtor, false), false) => {
                    trans_struct_drop(bcx, t, v0, dtor, did, substs)
                }
                (ty::NoDtor, _) | (_, true) => {
                    // No dtor? Just the default case
                    iter_structural_ty(bcx, v0, t, |bb, vv, tt| drop_ty(bb, vv, tt, DebugLoc::None))
                }
            }
        }
        ty::ty_trait(..) => {
            // No support in vtable for distinguishing destroying with
            // versus without calling Drop::drop. Assert caller is
            // okay with always calling the Drop impl, if any.
            assert!(!skip_dtor);
            let data_ptr = GEPi(bcx, v0, &[0, abi::FAT_PTR_ADDR]);
            let vtable_ptr = Load(bcx, GEPi(bcx, v0, &[0, abi::FAT_PTR_EXTRA]));
            let dtor = Load(bcx, vtable_ptr);
            Call(bcx,
                 dtor,
                 &[PointerCast(bcx, Load(bcx, data_ptr), Type::i8p(bcx.ccx()))],
                 None,
                 DebugLoc::None);
            bcx
        }
        _ => {
            if bcx.fcx.type_needs_drop(t) {
                iter_structural_ty(bcx,
                                   v0,
                                   t,
                                   |bb, vv, tt| drop_ty(bb, vv, tt, DebugLoc::None))
            } else {
                bcx
            }
        }
    }
}
