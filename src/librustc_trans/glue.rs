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
use std::ptr;
use std::iter;

use llvm;
use llvm::{ValueRef, get_param};
use middle::lang_items::BoxFreeFnLangItem;
use rustc::ty::subst::{Substs};
use rustc::traits;
use rustc::ty::{self, layout, AdtDef, AdtKind, Ty, TypeFoldable};
use rustc::ty::subst::Kind;
use rustc::mir::tcx::LvalueTy;
use mir::lvalue::LvalueRef;
use adt;
use base::*;
use callee::Callee;
use cleanup::CleanupScope;
use common::*;
use machine::*;
use monomorphize;
use trans_item::TransItem;
use tvec;
use type_of::{type_of, sizing_type_of, align_of};
use type_::Type;
use value::Value;
use Disr;
use builder::Builder;

use syntax_pos::DUMMY_SP;

pub fn trans_exchange_free_ty<'a, 'tcx>(bcx: &Builder<'a, 'tcx>, ptr: LvalueRef<'tcx>) {
    let content_ty = ptr.ty.to_ty(bcx.tcx());
    let def_id = langcall(bcx.tcx(), None, "", BoxFreeFnLangItem);
    let substs = bcx.tcx().mk_substs(iter::once(Kind::from(content_ty)));
    let callee = Callee::def(bcx.ccx, def_id, substs);

    let fn_ty = callee.direct_fn_type(bcx.ccx, &[]);

    let llret = bcx.call(callee.reify(bcx.ccx),
        &[ptr.llval, ptr.llextra][..1 + ptr.has_extra() as usize], None);
    fn_ty.apply_attrs_callsite(llret);
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
            scx.tcx().infer_ctxt((), traits::Reveal::All).enter(|infcx| {
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

fn drop_ty<'a, 'tcx>(bcx: &Builder<'a, 'tcx>, args: LvalueRef<'tcx>) {
    call_drop_glue(bcx, args, false, None)
}

pub fn call_drop_glue<'a, 'tcx>(
    bcx: &Builder<'a, 'tcx>,
    mut args: LvalueRef<'tcx>,
    skip_dtor: bool,
    funclet: Option<&'a Funclet>,
) {
    let t = args.ty.to_ty(bcx.tcx());
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
        if glue_type != t {
            args.llval = bcx.pointercast(args.llval, type_of(ccx, glue_type).ptr_to());
        }

        // No drop-hint ==> call standard drop glue
        bcx.call(glue, &[args.llval, args.llextra][..1 + args.has_extra() as usize],
            funclet.map(|b| b.bundle()));
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

    let mut bcx = Builder::new_block(ccx, llfn, "entry-block");

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
    let t = g.ty();

    let value = get_param(llfn, 0);
    let ptr = if ccx.shared().type_is_sized(t) {
        LvalueRef::new_sized_ty(value, t)
    } else {
        LvalueRef::new_unsized_ty(value, get_param(llfn, 1), t)
    };

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
            let ptr = if !bcx.ccx.shared().type_is_sized(content_ty) {
                let llbox = bcx.load(get_dataptr(&bcx, ptr.llval));
                let info = bcx.load(get_meta(&bcx, ptr.llval));
                LvalueRef::new_unsized_ty(llbox, info, content_ty)
            } else {
                LvalueRef::new_sized_ty(bcx.load(ptr.llval), content_ty)
            };
            drop_ty(&bcx, ptr);
            trans_exchange_free_ty(&bcx, ptr);
            bcx
        }
        ty::TyDynamic(..) => {
            // No support in vtable for distinguishing destroying with
            // versus without calling Drop::drop. Assert caller is
            // okay with always calling the Drop impl, if any.
            assert!(!skip_dtor);
            let dtor = bcx.load(ptr.llextra);
            bcx.call(dtor, &[ptr.llval], None);
            bcx
        }
        ty::TyAdt(def, ..) if def.dtor_kind().is_present() && !skip_dtor => {
            let shallow_drop = def.is_union();
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
                CleanupScope::schedule_drop_adt_contents(&bcx, ptr)
            } else {
                CleanupScope::noop()
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
            let args = &[ptr.llval, ptr.llextra][..1 + ptr.has_extra() as usize];
            if let Some(landing_pad) = contents_scope.landing_pad {
                let normal_bcx = bcx.build_sibling_block("normal-return");
                llret = bcx.invoke(callee.reify(ccx), args, normal_bcx.llbb(), landing_pad, None);
                bcx = normal_bcx;
            } else {
                llret = bcx.call(callee.reify(bcx.ccx), args, None);
            }
            fn_ty.apply_attrs_callsite(llret);
            contents_scope.trans(&bcx);
            bcx
        }
        ty::TyAdt(def, ..) if def.is_union() => {
            bcx
        }
        _ => {
            if bcx.ccx.shared().type_needs_drop(t) {
                drop_structural_ty(bcx, ptr)
            } else {
                bcx
            }
        }
    };
    bcx.ret_void();
}

pub fn size_and_align_of_dst<'a, 'tcx>(bcx: &Builder<'a, 'tcx>, t: Ty<'tcx>, info: ValueRef)
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
            let align = match (const_to_opt_u128(sized_align, false),
                               const_to_opt_u128(unsized_align, false)) {
                (Some(sized_align), Some(unsized_align)) => {
                    // If both alignments are constant, (the sized_align should always be), then
                    // pick the correct alignment statically.
                    C_uint(ccx, std::cmp::max(sized_align, unsized_align) as u64)
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
fn drop_structural_ty<'a, 'tcx>(
    cx: Builder<'a, 'tcx>,
    mut ptr: LvalueRef<'tcx>
) -> Builder<'a, 'tcx> {
    fn iter_variant_fields<'a, 'tcx>(
        cx: &'a Builder<'a, 'tcx>,
        av: LvalueRef<'tcx>,
        adt_def: &'tcx AdtDef,
        variant_index: usize,
        substs: &'tcx Substs<'tcx>
    ) {
        let variant = &adt_def.variants[variant_index];
        let tcx = cx.tcx();
        for (i, field) in variant.fields.iter().enumerate() {
            let arg = monomorphize::field_ty(tcx, substs, field);
            let field_ptr = av.trans_field_ptr(&cx, i);
            drop_ty(&cx, LvalueRef::new_sized_ty(field_ptr, arg));
        }
    }

    let mut cx = cx;
    let t = ptr.ty.to_ty(cx.tcx());
    match t.sty {
        ty::TyClosure(def_id, substs) => {
            for (i, upvar_ty) in substs.upvar_tys(def_id, cx.tcx()).enumerate() {
                let llupvar = ptr.trans_field_ptr(&cx, i);
                drop_ty(&cx, LvalueRef::new_sized_ty(llupvar, upvar_ty));
            }
        }
        ty::TyArray(_, n) => {
            let base = get_dataptr(&cx, ptr.llval);
            let len = C_uint(cx.ccx, n);
            let unit_ty = t.sequence_element_type(cx.tcx());
            cx = tvec::slice_for_each(&cx, base, unit_ty, len,
                |bb, vv| drop_ty(bb, LvalueRef::new_sized_ty(vv, unit_ty)));
        }
        ty::TySlice(_) | ty::TyStr => {
            let unit_ty = t.sequence_element_type(cx.tcx());
            cx = tvec::slice_for_each(&cx, ptr.llval, unit_ty, ptr.llextra,
                |bb, vv| drop_ty(bb, LvalueRef::new_sized_ty(vv, unit_ty)));
        }
        ty::TyTuple(ref args) => {
            for (i, arg) in args.iter().enumerate() {
                let llfld_a = ptr.trans_field_ptr(&cx, i);
                drop_ty(&cx, LvalueRef::new_sized_ty(llfld_a, *arg));
            }
        }
        ty::TyAdt(adt, substs) => match adt.adt_kind() {
            AdtKind::Struct => {
                for (i, field) in adt.variants[0].fields.iter().enumerate() {
                    let field_ty = monomorphize::field_ty(cx.tcx(), substs, field);
                    let mut field_ptr = ptr.clone();
                    field_ptr.llval = ptr.trans_field_ptr(&cx, i);
                    field_ptr.ty = LvalueTy::from_ty(field_ty);
                    if cx.ccx.shared().type_is_sized(field_ty) {
                        field_ptr.llextra = ptr::null_mut();
                    }
                    drop_ty(&cx, field_ptr);
                }
            }
            AdtKind::Union => {
                bug!("Union in `glue::drop_structural_ty`");
            }
            AdtKind::Enum => {
                let n_variants = adt.variants.len();

                // NB: we must hit the discriminant first so that structural
                // comparison know not to proceed when the discriminants differ.

                // Obtain a representation of the discriminant sufficient to translate
                // destructuring; this may or may not involve the actual discriminant.
                let l = cx.ccx.layout_of(t);
                match *l {
                    layout::Univariant { .. } |
                    layout::UntaggedUnion { .. } => {
                        if n_variants != 0 {
                            assert!(n_variants == 1);
                            ptr.ty = LvalueTy::Downcast {
                                adt_def: adt,
                                substs: substs,
                                variant_index: 0,
                            };
                            iter_variant_fields(&cx, ptr, &adt, 0, substs);
                        }
                    }
                    layout::CEnum { .. } |
                    layout::General { .. } |
                    layout::RawNullablePointer { .. } |
                    layout::StructWrappedNullablePointer { .. } => {
                        let lldiscrim_a = adt::trans_get_discr(&cx, t, ptr.llval, None, false);
                        let tcx = cx.tcx();
                        drop_ty(&cx, LvalueRef::new_sized_ty(lldiscrim_a, tcx.types.isize));

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
                        let ret_void_cx = cx.build_sibling_block("enum-iter-ret-void");
                        ret_void_cx.ret_void();
                        let llswitch = cx.switch(lldiscrim_a, ret_void_cx.llbb(), n_variants);
                        let next_cx = cx.build_sibling_block("enum-iter-next");

                        for (i, variant) in adt.variants.iter().enumerate() {
                            let variant_cx_name = format!("enum-iter-variant-{}",
                                &variant.disr_val.to_string());
                            let variant_cx = cx.build_sibling_block(&variant_cx_name);
                            let case_val = adt::trans_case(&cx, t, Disr::from(variant.disr_val));
                            variant_cx.add_case(llswitch, case_val, variant_cx.llbb());
                            ptr.ty = LvalueTy::Downcast {
                                adt_def: adt,
                                substs: substs,
                                variant_index: i,
                            };
                            iter_variant_fields(&variant_cx, ptr, &adt, i, substs);
                            variant_cx.br(next_cx.llbb());
                        }
                        cx = next_cx;
                    }
                    _ => bug!("{} is not an enum.", t),
                }
            }
        },

        _ => {
            cx.sess().unimpl(&format!("type in drop_structural_ty: {}", t))
        }
    }
    return cx;
}
