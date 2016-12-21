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

use std;

use llvm;
use llvm::{ValueRef, get_param};
use middle::lang_items::ExchangeFreeFnLangItem;
use rustc::ty::subst::{Substs};
use rustc::traits;
use rustc::ty::{self, AdtKind, Ty, TypeFoldable};
use adt;
use base::*;
use callee::Callee;
use common::*;
use machine::*;
use monomorphize;
use trans_item::TransItem;
use tvec;
use type_of::{type_of, sizing_type_of, align_of};
use type_::Type;
use value::Value;
use Disr;
use cleanup::CleanupScope;

use syntax_pos::DUMMY_SP;

pub fn trans_exchange_free_dyn<'a, 'tcx>(
    bcx: &BlockAndBuilder<'a, 'tcx>,
    v: ValueRef,
    size: ValueRef,
    align: ValueRef
) {
    let def_id = langcall(bcx.tcx(), None, "", ExchangeFreeFnLangItem);
    let args = [bcx.pointercast(v, Type::i8p(bcx.ccx)), size, align];
    let callee = Callee::def(bcx.ccx, def_id, bcx.tcx().intern_substs(&[]));

    let ccx = bcx.ccx;
    let fn_ty = callee.direct_fn_type(ccx, &[]);

    let llret = bcx.call(callee.reify(ccx), &args[..], None);
    fn_ty.apply_attrs_callsite(llret);
}

pub fn trans_exchange_free_ty<'a, 'tcx>(
    bcx: &BlockAndBuilder<'a, 'tcx>, ptr: ValueRef, content_ty: Ty<'tcx>
) {
    assert!(bcx.ccx.shared().type_is_sized(content_ty));
    let sizing_type = sizing_type_of(bcx.ccx, content_ty);
    let content_size = llsize_of_alloc(bcx.ccx, sizing_type);

    // `Box<ZeroSizeType>` does not allocate.
    if content_size != 0 {
        let content_align = align_of(bcx.ccx, content_ty);
        let ccx = bcx.ccx;
        trans_exchange_free_dyn(bcx, ptr, C_uint(ccx, content_size), C_uint(ccx, content_align));
    }
}

pub fn get_drop_glue_type<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>, t: Ty<'tcx>) -> Ty<'tcx> {
    assert!(t.is_normalized_for_trans());

    let t = scx.tcx().erase_regions(&t);

    // Even if there is no dtor for t, there might be one deeper down and we
    // might need to pass in the vtable ptr.
    if !scx.type_is_sized(t) {
        return t;
    }

    // FIXME (#22815): note that type_needs_drop conservatively
    // approximates in some cases and may say a type expression
    // requires drop glue when it actually does not.
    //
    // (In this case it is not clear whether any harm is done, i.e.
    // erroneously returning `t` in some cases where we could have
    // returned `tcx.types.i8` does not appear unsound. The impact on
    // code quality is unknown at this time.)

    if !scx.type_needs_drop(t) {
        return scx.tcx().types.i8;
    }
    match t.sty {
        ty::TyBox(typ) if !scx.type_needs_drop(typ) && scx.type_is_sized(typ) => {
            scx.tcx().infer_ctxt(None, None, traits::Reveal::All).enter(|infcx| {
                let layout = t.layout(&infcx).unwrap();
                if layout.size(&scx.tcx().data_layout).bytes() == 0 {
                    // `Box<ZeroSizeType>` does not allocate.
                    scx.tcx().types.i8
                } else {
                    t
                }
            })
        }
        _ => t
    }
}

fn drop_ty<'a, 'tcx>(bcx: &BlockAndBuilder<'a, 'tcx>, v: ValueRef, t: Ty<'tcx>) {
    call_drop_glue(bcx, v, t, false, None)
}

pub fn call_drop_glue<'a, 'tcx>(
    bcx: &BlockAndBuilder<'a, 'tcx>,
    v: ValueRef,
    t: Ty<'tcx>,
    skip_dtor: bool,
    funclet: Option<&'a Funclet>,
) {
    // NB: v is an *alias* of type t here, not a direct value.
    debug!("call_drop_glue(t={:?}, skip_dtor={})", t, skip_dtor);
    if bcx.ccx.shared().type_needs_drop(t) {
        let ccx = bcx.ccx;
        let g = if skip_dtor {
            DropGlueKind::TyContents(t)
        } else {
            DropGlueKind::Ty(t)
        };
        let glue = get_drop_glue_core(ccx, g);
        let glue_type = get_drop_glue_type(ccx.shared(), t);
        let ptr = if glue_type != t {
            bcx.pointercast(v, type_of(ccx, glue_type).ptr_to())
        } else {
            v
        };

        // No drop-hint ==> call standard drop glue
        bcx.call(glue, &[ptr], funclet.map(|b| b.bundle()));
    }
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
    /// if Ty is Newtype(S) then only the Drop impl for Newtype itself
    /// will be skipped, while the Drop impl for S, if any, will be
    /// invoked.
    TyContents(Ty<'tcx>),
}

impl<'tcx> DropGlueKind<'tcx> {
    pub fn ty(&self) -> Ty<'tcx> {
        match *self { DropGlueKind::Ty(t) | DropGlueKind::TyContents(t) => t }
    }

    pub fn map_ty<F>(&self, mut f: F) -> DropGlueKind<'tcx> where F: FnMut(Ty<'tcx>) -> Ty<'tcx>
    {
        match *self {
            DropGlueKind::Ty(t) => DropGlueKind::Ty(f(t)),
            DropGlueKind::TyContents(t) => DropGlueKind::TyContents(f(t)),
        }
    }
}

fn get_drop_glue_core<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, g: DropGlueKind<'tcx>) -> ValueRef {
    let g = g.map_ty(|t| get_drop_glue_type(ccx.shared(), t));
    match ccx.drop_glues().borrow().get(&g) {
        Some(&(glue, _)) => glue,
        None => {
            bug!("Could not find drop glue for {:?} -- {} -- {}.",
                    g,
                    TransItem::DropGlue(g).to_raw_string(),
                    ccx.codegen_unit().name());
        }
    }
}

pub fn implement_drop_glue<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, g: DropGlueKind<'tcx>) {
    assert_eq!(g.ty(), get_drop_glue_type(ccx.shared(), g.ty()));
    let (llfn, _) = ccx.drop_glues().borrow().get(&g).unwrap().clone();

    let fcx = FunctionContext::new(ccx, llfn);
    let bcx = fcx.get_entry_block();

    ccx.stats().n_glues_created.set(ccx.stats().n_glues_created.get() + 1);
    // All glue functions take values passed *by alias*; this is a
    // requirement since in many contexts glue is invoked indirectly and
    // the caller has no idea if it's dealing with something that can be
    // passed by value.
    //
    // llfn is expected be declared to take a parameter of the appropriate
    // type, so we don't need to explicitly cast the function parameter.

    // NB: v0 is an *alias* of type t here, not a direct value.
    // Only drop the value when it ... well, we used to check for
    // non-null, (and maybe we need to continue doing so), but we now
    // must definitely check for special bit-patterns corresponding to
    // the special dtor markings.
    let v0 = get_param(llfn, 0);
    let t = g.ty();

    let skip_dtor = match g {
        DropGlueKind::Ty(_) => false,
        DropGlueKind::TyContents(_) => true
    };

    let bcx = match t.sty {
        ty::TyBox(content_ty) => {
            // Support for TyBox is built-in and its drop glue is
            // special. It may move to library and have Drop impl. As
            // a safe-guard, assert TyBox not used with TyContents.
            assert!(!skip_dtor);
            if !bcx.ccx.shared().type_is_sized(content_ty) {
                let llval = get_dataptr(&bcx, v0);
                let llbox = bcx.load(llval);
                drop_ty(&bcx, v0, content_ty);
                // FIXME(#36457) -- we should pass unsized values to drop glue as two arguments
                let info = get_meta(&bcx, v0);
                let info = bcx.load(info);
                let (llsize, llalign) = size_and_align_of_dst(&bcx, content_ty, info);

                // `Box<ZeroSizeType>` does not allocate.
                let needs_free = bcx.icmp(llvm::IntNE, llsize, C_uint(bcx.ccx, 0u64));
                if const_to_opt_uint(needs_free) == Some(0) {
                    bcx
                } else {
                    let next_cx = bcx.fcx().build_new_block("next");
                    let cond_cx = bcx.fcx().build_new_block("cond");
                    bcx.cond_br(needs_free, cond_cx.llbb(), next_cx.llbb());
                    trans_exchange_free_dyn(&cond_cx, llbox, llsize, llalign);
                    cond_cx.br(next_cx.llbb());
                    next_cx
                }
            } else {
                let llval = v0;
                let llbox = bcx.load(llval);
                drop_ty(&bcx, llbox, content_ty);
                trans_exchange_free_ty(&bcx, llbox, content_ty);
                bcx
            }
        }
        ty::TyDynamic(..) => {
            // No support in vtable for distinguishing destroying with
            // versus without calling Drop::drop. Assert caller is
            // okay with always calling the Drop impl, if any.
            // FIXME(#36457) -- we should pass unsized values to drop glue as two arguments
            assert!(!skip_dtor);
            let data_ptr = get_dataptr(&bcx, v0);
            let vtable_ptr = bcx.load(get_meta(&bcx, v0));
            let dtor = bcx.load(vtable_ptr);
            bcx.call(dtor, &[bcx.pointercast(bcx.load(data_ptr), Type::i8p(bcx.ccx))], None);
            bcx
        }
        ty::TyAdt(def, ..) if def.dtor_kind().is_present() && !skip_dtor => {
            trans_custom_dtor(bcx, t, v0, def.is_union())
        }
        ty::TyAdt(def, ..) if def.is_union() => {
            bcx
        }
        _ => {
            if bcx.ccx.shared().type_needs_drop(t) {
                drop_structural_ty(bcx, v0, t)
            } else {
                bcx
            }
        }
    };
    bcx.ret_void();
}

fn trans_custom_dtor<'a, 'tcx>(mut bcx: BlockAndBuilder<'a, 'tcx>,
                               t: Ty<'tcx>,
                               v0: ValueRef,
                               shallow_drop: bool)
                               -> BlockAndBuilder<'a, 'tcx>
{
    debug!("trans_custom_dtor t: {}", t);
    let tcx = bcx.tcx();

    let def = t.ty_adt_def().unwrap();

    // Be sure to put the contents into a scope so we can use an invoke
    // instruction to call the user destructor but still call the field
    // destructors if the user destructor panics.
    //
    // FIXME (#14875) panic-in-drop semantics might be unsupported; we
    // might well consider changing below to more direct code.
    // Issue #23611: schedule cleanup of contents, re-inspecting the
    // discriminant (if any) in case of variant swap in drop code.
    let contents_scope = if !shallow_drop {
        bcx.fcx().schedule_drop_adt_contents(v0, t)
    } else {
        CleanupScope::noop()
    };

    let (sized_args, unsized_args);
    let args: &[ValueRef] = if bcx.ccx.shared().type_is_sized(t) {
        sized_args = [v0];
        &sized_args
    } else {
        // FIXME(#36457) -- we should pass unsized values to drop glue as two arguments
        unsized_args = [
            bcx.load(get_dataptr(&bcx, v0)),
            bcx.load(get_meta(&bcx, v0))
        ];
        &unsized_args
    };

    let trait_ref = ty::Binder(ty::TraitRef {
        def_id: tcx.lang_items.drop_trait().unwrap(),
        substs: tcx.mk_substs_trait(t, &[])
    });
    let vtbl = match fulfill_obligation(bcx.ccx.shared(), DUMMY_SP, trait_ref) {
        traits::VtableImpl(data) => data,
        _ => bug!("dtor for {:?} is not an impl???", t)
    };
    let dtor_did = def.destructor().unwrap();
    let callee = Callee::def(bcx.ccx, dtor_did, vtbl.substs);
    let fn_ty = callee.direct_fn_type(bcx.ccx, &[]);
    let llret;
    if let Some(landing_pad) = contents_scope.landing_pad {
        let normal_bcx = bcx.fcx().build_new_block("normal-return");
        llret = bcx.invoke(callee.reify(bcx.ccx), args, normal_bcx.llbb(), landing_pad, None);
        bcx = normal_bcx;
    } else {
        llret = bcx.call(callee.reify(bcx.ccx), args, None);
    }
    fn_ty.apply_attrs_callsite(llret);
    contents_scope.trans(&bcx);
    bcx
}

pub fn size_and_align_of_dst<'a, 'tcx>(bcx: &BlockAndBuilder<'a, 'tcx>,
                                       t: Ty<'tcx>, info: ValueRef)
                                       -> (ValueRef, ValueRef) {
    debug!("calculate size of DST: {}; with lost info: {:?}",
           t, Value(info));
    if bcx.ccx.shared().type_is_sized(t) {
        let sizing_type = sizing_type_of(bcx.ccx, t);
        let size = llsize_of_alloc(bcx.ccx, sizing_type);
        let align = align_of(bcx.ccx, t);
        debug!("size_and_align_of_dst t={} info={:?} size: {} align: {}",
               t, Value(info), size, align);
        let size = C_uint(bcx.ccx, size);
        let align = C_uint(bcx.ccx, align);
        return (size, align);
    }
    match t.sty {
        ty::TyAdt(def, substs) => {
            let ccx = bcx.ccx;
            // First get the size of all statically known fields.
            // Don't use type_of::sizing_type_of because that expects t to be sized,
            // and it also rounds up to alignment, which we want to avoid,
            // as the unsized field's alignment could be smaller.
            assert!(!t.is_simd());
            let layout = ccx.layout_of(t);
            debug!("DST {} layout: {:?}", t, layout);

            let (sized_size, sized_align) = match *layout {
                ty::layout::Layout::Univariant { ref variant, .. } => {
                    (variant.offsets.last().map_or(0, |o| o.bytes()), variant.align.abi())
                }
                _ => {
                    bug!("size_and_align_of_dst: expcted Univariant for `{}`, found {:#?}",
                         t, layout);
                }
            };
            debug!("DST {} statically sized prefix size: {} align: {}",
                   t, sized_size, sized_align);
            let sized_size = C_uint(ccx, sized_size);
            let sized_align = C_uint(ccx, sized_align);

            // Recurse to get the size of the dynamically sized field (must be
            // the last field).
            let last_field = def.struct_variant().fields.last().unwrap();
            let field_ty = monomorphize::field_ty(bcx.tcx(), substs, last_field);
            let (unsized_size, unsized_align) = size_and_align_of_dst(bcx, field_ty, info);

            // FIXME (#26403, #27023): We should be adding padding
            // to `sized_size` (to accommodate the `unsized_align`
            // required of the unsized field that follows) before
            // summing it with `sized_size`. (Note that since #26403
            // is unfixed, we do not yet add the necessary padding
            // here. But this is where the add would go.)

            // Return the sum of sizes and max of aligns.
            let size = bcx.add(sized_size, unsized_size);

            // Choose max of two known alignments (combined value must
            // be aligned according to more restrictive of the two).
            let align = match (const_to_opt_uint(sized_align), const_to_opt_uint(unsized_align)) {
                (Some(sized_align), Some(unsized_align)) => {
                    // If both alignments are constant, (the sized_align should always be), then
                    // pick the correct alignment statically.
                    C_uint(ccx, std::cmp::max(sized_align, unsized_align))
                }
                _ => bcx.select(bcx.icmp(llvm::IntUGT, sized_align, unsized_align),
                                sized_align,
                                unsized_align)
            };

            // Issue #27023: must add any necessary padding to `size`
            // (to make it a multiple of `align`) before returning it.
            //
            // Namely, the returned size should be, in C notation:
            //
            //   `size + ((size & (align-1)) ? align : 0)`
            //
            // emulated via the semi-standard fast bit trick:
            //
            //   `(size + (align-1)) & -align`

            let addend = bcx.sub(align, C_uint(bcx.ccx, 1_u64));
            let size = bcx.and(bcx.add(size, addend), bcx.neg(align));

            (size, align)
        }
        ty::TyDynamic(..) => {
            // info points to the vtable and the second entry in the vtable is the
            // dynamic size of the object.
            let info = bcx.pointercast(info, Type::int(bcx.ccx).ptr_to());
            let size_ptr = bcx.gepi(info, &[1]);
            let align_ptr = bcx.gepi(info, &[2]);
            (bcx.load(size_ptr), bcx.load(align_ptr))
        }
        ty::TySlice(_) | ty::TyStr => {
            let unit_ty = t.sequence_element_type(bcx.tcx());
            // The info in this case is the length of the str, so the size is that
            // times the unit size.
            let llunit_ty = sizing_type_of(bcx.ccx, unit_ty);
            let unit_align = llalign_of_min(bcx.ccx, llunit_ty);
            let unit_size = llsize_of_alloc(bcx.ccx, llunit_ty);
            (bcx.mul(info, C_uint(bcx.ccx, unit_size)),
             C_uint(bcx.ccx, unit_align))
        }
        _ => bug!("Unexpected unsized type, found {}", t)
    }
}

// Iterates through the elements of a structural type, dropping them.
fn drop_structural_ty<'a, 'tcx>(cx: BlockAndBuilder<'a, 'tcx>,
                                av: ValueRef,
                                t: Ty<'tcx>)
                                -> BlockAndBuilder<'a, 'tcx> {
    fn iter_variant<'a, 'tcx>(cx: &BlockAndBuilder<'a, 'tcx>,
                              t: Ty<'tcx>,
                              av: adt::MaybeSizedValue,
                              variant: &'tcx ty::VariantDef,
                              substs: &Substs<'tcx>) {
        let tcx = cx.tcx();
        for (i, field) in variant.fields.iter().enumerate() {
            let arg = monomorphize::field_ty(tcx, substs, field);
            let field_ptr = adt::trans_field_ptr(&cx, t, av, Disr::from(variant.disr_val), i);
            drop_ty(&cx, field_ptr, arg);
        }
    }

    let value = if cx.ccx.shared().type_is_sized(t) {
        adt::MaybeSizedValue::sized(av)
    } else {
        // FIXME(#36457) -- we should pass unsized values as two arguments
        let data = cx.load(get_dataptr(&cx, av));
        let info = cx.load(get_meta(&cx, av));
        adt::MaybeSizedValue::unsized_(data, info)
    };

    let mut cx = cx;
    match t.sty {
        ty::TyClosure(def_id, substs) => {
            for (i, upvar_ty) in substs.upvar_tys(def_id, cx.tcx()).enumerate() {
                let llupvar = adt::trans_field_ptr(&cx, t, value, Disr(0), i);
                drop_ty(&cx, llupvar, upvar_ty);
            }
        }
        ty::TyArray(_, n) => {
            let base = get_dataptr(&cx, value.value);
            let len = C_uint(cx.ccx, n);
            let unit_ty = t.sequence_element_type(cx.tcx());
            cx = tvec::slice_for_each(&cx, base, unit_ty, len, |bb, vv| drop_ty(bb, vv, unit_ty));
        }
        ty::TySlice(_) | ty::TyStr => {
            let unit_ty = t.sequence_element_type(cx.tcx());
            cx = tvec::slice_for_each(&cx, value.value, unit_ty, value.meta,
                |bb, vv| drop_ty(bb, vv, unit_ty));
        }
        ty::TyTuple(ref args) => {
            for (i, arg) in args.iter().enumerate() {
                let llfld_a = adt::trans_field_ptr(&cx, t, value, Disr(0), i);
                drop_ty(&cx, llfld_a, *arg);
            }
        }
        ty::TyAdt(adt, substs) => match adt.adt_kind() {
            AdtKind::Struct => {
                let VariantInfo { fields, discr } = VariantInfo::from_ty(cx.tcx(), t, None);
                for (i, &Field(_, field_ty)) in fields.iter().enumerate() {
                    let llfld_a = adt::trans_field_ptr(&cx, t, value, Disr::from(discr), i);

                    let val = if cx.ccx.shared().type_is_sized(field_ty) {
                        llfld_a
                    } else {
                        // FIXME(#36457) -- we should pass unsized values as two arguments
                        let scratch = alloc_ty(&cx, field_ty, "__fat_ptr_iter");
                        cx.store(llfld_a, get_dataptr(&cx, scratch));
                        cx.store(value.meta, get_meta(&cx, scratch));
                        scratch
                    };
                    drop_ty(&cx, val, field_ty);
                }
            }
            AdtKind::Union => {
                bug!("Union in `glue::drop_structural_ty`");
            }
            AdtKind::Enum => {
                let n_variants = adt.variants.len();

                // NB: we must hit the discriminant first so that structural
                // comparison know not to proceed when the discriminants differ.

                match adt::trans_switch(&cx, t, av, false) {
                    (adt::BranchKind::Single, None) => {
                        if n_variants != 0 {
                            assert!(n_variants == 1);
                            iter_variant(&cx, t, adt::MaybeSizedValue::sized(av),
                                            &adt.variants[0], substs);
                        }
                    }
                    (adt::BranchKind::Switch, Some(lldiscrim_a)) => {
                        let tcx = cx.tcx();
                        drop_ty(&cx, lldiscrim_a, tcx.types.isize);

                        // Create a fall-through basic block for the "else" case of
                        // the switch instruction we're about to generate. Note that
                        // we do **not** use an Unreachable instruction here, even
                        // though most of the time this basic block will never be hit.
                        //
                        // When an enum is dropped it's contents are currently
                        // overwritten to DTOR_DONE, which means the discriminant
                        // could have changed value to something not within the actual
                        // range of the discriminant. Currently this function is only
                        // used for drop glue so in this case we just return quickly
                        // from the outer function, and any other use case will only
                        // call this for an already-valid enum in which case the `ret
                        // void` will never be hit.
                        let ret_void_cx = cx.fcx().build_new_block("enum-iter-ret-void");
                        ret_void_cx.ret_void();
                        let llswitch = cx.switch(lldiscrim_a, ret_void_cx.llbb(), n_variants);
                        let next_cx = cx.fcx().build_new_block("enum-iter-next");

                        for variant in &adt.variants {
                            let variant_cx_name = format!("enum-iter-variant-{}",
                                &variant.disr_val.to_string());
                            let variant_cx = cx.fcx().build_new_block(&variant_cx_name);
                            let case_val = adt::trans_case(&cx, t, Disr::from(variant.disr_val));
                            variant_cx.add_case(llswitch, case_val, variant_cx.llbb());
                            iter_variant(&variant_cx, t, value, variant, substs);
                            variant_cx.br(next_cx.llbb());
                        }
                        cx = next_cx;
                    }
                    _ => cx.ccx.sess().unimpl("value from adt::trans_switch in drop_structural_ty"),
                }
            }
        },

        _ => {
            cx.sess().unimpl(&format!("type in drop_structural_ty: {}", t))
        }
    }
    return cx;
}
