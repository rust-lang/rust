// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{self, ValueRef};
use rustc::ty::{self, Ty, TypeFoldable};
use rustc::ty::layout::{self, Align, LayoutTyper};
use rustc::mir;
use rustc::mir::tcx::LvalueTy;
use rustc_data_structures::indexed_vec::Idx;
use adt;
use base;
use builder::Builder;
use common::{self, CrateContext, C_usize, C_u8, C_i32, C_int, C_null, val_ty};
use consts;
use type_of;
use type_::Type;
use value::Value;
use glue;

use std::ptr;
use std::ops;

use super::{MirContext, LocalRef};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Alignment {
    Packed,
    AbiAligned,
}

impl ops::BitOr for Alignment {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Alignment::Packed, _) => Alignment::Packed,
            (Alignment::AbiAligned, a) => a,
        }
    }
}

impl Alignment {
    pub fn from_packed(packed: bool) -> Self {
        if packed {
            Alignment::Packed
        } else {
            Alignment::AbiAligned
        }
    }

    pub fn to_align(self) -> Option<Align> {
        match self {
            Alignment::Packed => Some(Align::from_bytes(1, 1).unwrap()),
            Alignment::AbiAligned => None,
        }
    }

    pub fn min_with(self, align: Option<Align>) -> Option<Align> {
        self.to_align().or(align)
    }
}

fn target_sets_discr_via_memset<'a, 'tcx>(bcx: &Builder<'a, 'tcx>) -> bool {
    bcx.sess().target.target.arch == "arm" || bcx.sess().target.target.arch == "aarch64"
}

#[derive(Copy, Clone, Debug)]
pub struct LvalueRef<'tcx> {
    /// Pointer to the contents of the lvalue
    pub llval: ValueRef,

    /// This lvalue's extra data if it is unsized, or null
    pub llextra: ValueRef,

    /// Monomorphized type of this lvalue, including variant information
    pub ty: LvalueTy<'tcx>,

    /// Whether this lvalue is known to be aligned according to its layout
    pub alignment: Alignment,
}

impl<'a, 'tcx> LvalueRef<'tcx> {
    pub fn new_sized(llval: ValueRef, lvalue_ty: LvalueTy<'tcx>,
                     alignment: Alignment) -> LvalueRef<'tcx> {
        LvalueRef { llval: llval, llextra: ptr::null_mut(), ty: lvalue_ty, alignment: alignment }
    }

    pub fn new_sized_ty(llval: ValueRef, ty: Ty<'tcx>, alignment: Alignment) -> LvalueRef<'tcx> {
        LvalueRef::new_sized(llval, LvalueTy::from_ty(ty), alignment)
    }

    pub fn alloca(bcx: &Builder<'a, 'tcx>, ty: Ty<'tcx>, name: &str) -> LvalueRef<'tcx> {
        debug!("alloca({:?}: {:?})", name, ty);
        let tmp = bcx.alloca(
            type_of::type_of(bcx.ccx, ty), name, bcx.ccx.over_align_of(ty));
        assert!(!ty.has_param_types());
        Self::new_sized_ty(tmp, ty, Alignment::AbiAligned)
    }

    pub fn len(&self, ccx: &CrateContext<'a, 'tcx>) -> ValueRef {
        let ty = self.ty.to_ty(ccx.tcx());
        match ty.sty {
            ty::TyArray(_, n) => {
                common::C_usize(ccx, n.val.to_const_int().unwrap().to_u64().unwrap())
            }
            ty::TySlice(_) | ty::TyStr => {
                assert!(self.llextra != ptr::null_mut());
                self.llextra
            }
            _ => bug!("unexpected type `{}` in LvalueRef::len", ty)
        }
    }

    pub fn has_extra(&self) -> bool {
        !self.llextra.is_null()
    }

    /// Access a field, at a point when the value's case is known.
    pub fn trans_field_ptr(self, bcx: &Builder<'a, 'tcx>, ix: usize) -> (ValueRef, Alignment) {
        let ccx = bcx.ccx;
        let mut l = ccx.layout_of(self.ty.to_ty(bcx.tcx()));
        match self.ty {
            LvalueTy::Ty { .. } => {}
            LvalueTy::Downcast { variant_index, .. } => {
                l = l.for_variant(variant_index)
            }
        }
        let fty = l.field(ccx, ix).ty;
        let mut ix = ix;
        let st = match *l {
            layout::Vector { .. } => {
                return (bcx.struct_gep(self.llval, ix), self.alignment);
            }
            layout::UntaggedUnion { ref variants } => {
                let ty = type_of::in_memory_type_of(ccx, fty);
                return (bcx.pointercast(self.llval, ty.ptr_to()),
                    self.alignment | Alignment::from_packed(variants.packed));
            }
            layout::RawNullablePointer { nndiscr, .. } |
            layout::StructWrappedNullablePointer { nndiscr,  .. }
                if l.variant_index.unwrap() as u64 != nndiscr => {
                // The unit-like case might have a nonzero number of unit-like fields.
                // (e.d., Result of Either with (), as one side.)
                let ty = type_of::type_of(ccx, fty);
                assert_eq!(ccx.size_of(fty).bytes(), 0);
                return (bcx.pointercast(self.llval, ty.ptr_to()), Alignment::Packed);
            }
            layout::RawNullablePointer { .. } => {
                let ty = type_of::type_of(ccx, fty);
                return (bcx.pointercast(self.llval, ty.ptr_to()), self.alignment);
            }
            layout::Univariant { ref variant, .. } => variant,
            layout::StructWrappedNullablePointer { ref nonnull, .. } => nonnull,
            layout::General { ref variants, .. } => {
                ix += 1;
                &variants[l.variant_index.unwrap()]
            }
            _ => bug!("element access in type without elements: {} represented as {:#?}", l.ty, l)
        };

        let alignment = self.alignment | Alignment::from_packed(st.packed);

        let ptr_val = if let layout::General { discr, .. } = *l {
            let variant_ty = Type::struct_(ccx,
                &adt::struct_llfields(ccx, l.ty, l.variant_index.unwrap(), st,
                                      Some(discr.to_ty(bcx.tcx(), false))), st.packed);
            bcx.pointercast(self.llval, variant_ty.ptr_to())
        } else {
            self.llval
        };

        // Simple case - we can just GEP the field
        //   * First field - Always aligned properly
        //   * Packed struct - There is no alignment padding
        //   * Field is sized - pointer is properly aligned already
        if st.offsets[ix] == layout::Size::from_bytes(0) || st.packed ||
            ccx.shared().type_is_sized(fty)
        {
            return (bcx.struct_gep(
                    ptr_val, adt::struct_llfields_index(st, ix)), alignment);
        }

        // If the type of the last field is [T], str or a foreign type, then we don't need to do
        // any adjusments
        match fty.sty {
            ty::TySlice(..) | ty::TyStr | ty::TyForeign(..) => {
                return (bcx.struct_gep(
                        ptr_val, adt::struct_llfields_index(st, ix)), alignment);
            }
            _ => ()
        }

        // There's no metadata available, log the case and just do the GEP.
        if !self.has_extra() {
            debug!("Unsized field `{}`, of `{:?}` has no metadata for adjustment",
                ix, Value(ptr_val));
            return (bcx.struct_gep(ptr_val, adt::struct_llfields_index(st, ix)), alignment);
        }

        // We need to get the pointer manually now.
        // We do this by casting to a *i8, then offsetting it by the appropriate amount.
        // We do this instead of, say, simply adjusting the pointer from the result of a GEP
        // because the field may have an arbitrary alignment in the LLVM representation
        // anyway.
        //
        // To demonstrate:
        //   struct Foo<T: ?Sized> {
        //      x: u16,
        //      y: T
        //   }
        //
        // The type Foo<Foo<Trait>> is represented in LLVM as { u16, { u16, u8 }}, meaning that
        // the `y` field has 16-bit alignment.

        let meta = self.llextra;


        let offset = st.offsets[ix].bytes();
        let unaligned_offset = C_usize(ccx, offset);

        // Get the alignment of the field
        let (_, align) = glue::size_and_align_of_dst(bcx, fty, meta);

        // Bump the unaligned offset up to the appropriate alignment using the
        // following expression:
        //
        //   (unaligned offset + (align - 1)) & -align

        // Calculate offset
        let align_sub_1 = bcx.sub(align, C_usize(ccx, 1u64));
        let offset = bcx.and(bcx.add(unaligned_offset, align_sub_1),
        bcx.neg(align));

        debug!("struct_field_ptr: DST field offset: {:?}", Value(offset));

        // Cast and adjust pointer
        let byte_ptr = bcx.pointercast(ptr_val, Type::i8p(ccx));
        let byte_ptr = bcx.gep(byte_ptr, &[offset]);

        // Finally, cast back to the type expected
        let ll_fty = type_of::in_memory_type_of(ccx, fty);
        debug!("struct_field_ptr: Field type is {:?}", ll_fty);
        (bcx.pointercast(byte_ptr, ll_fty.ptr_to()), alignment)
    }

    // Double index to account for padding (FieldPath already uses `Struct::memory_index`)
    fn gepi_struct_llfields_path(self, bcx: &Builder, discrfield: &layout::FieldPath) -> ValueRef {
        let path = discrfield.iter().map(|&i| (i as usize) << 1).collect::<Vec<_>>();
        bcx.gepi(self.llval, &path)
    }

    /// Helper for cases where the discriminant is simply loaded.
    fn load_discr(self, bcx: &Builder, ity: layout::Integer, ptr: ValueRef,
                  min: u64, max: u64) -> ValueRef {
        let llty = Type::from_integer(bcx.ccx, ity);
        assert_eq!(val_ty(ptr), llty.ptr_to());
        let bits = ity.size().bits();
        assert!(bits <= 64);
        let bits = bits as usize;
        let mask = !0u64 >> (64 - bits);
        // For a (max) discr of -1, max will be `-1 as usize`, which overflows.
        // However, that is fine here (it would still represent the full range),
        if max.wrapping_add(1) & mask == min & mask {
            // i.e., if the range is everything.  The lo==hi case would be
            // rejected by the LLVM verifier (it would mean either an
            // empty set, which is impossible, or the entire range of the
            // type, which is pointless).
            bcx.load(ptr, self.alignment.to_align())
        } else {
            // llvm::ConstantRange can deal with ranges that wrap around,
            // so an overflow on (max + 1) is fine.
            bcx.load_range_assert(ptr, min, max.wrapping_add(1), /* signed: */ llvm::True,
                                  self.alignment.to_align())
        }
    }

    /// Obtain the actual discriminant of a value.
    pub fn trans_get_discr(self, bcx: &Builder<'a, 'tcx>, cast_to: Ty<'tcx>) -> ValueRef {
        let l = bcx.ccx.layout_of(self.ty.to_ty(bcx.tcx()));

        let val = match *l {
            layout::CEnum { discr, min, max, .. } => {
                self.load_discr(bcx, discr, self.llval, min, max)
            }
            layout::General { discr, ref variants, .. } => {
                let ptr = bcx.struct_gep(self.llval, 0);
                self.load_discr(bcx, discr, ptr, 0, variants.len() as u64 - 1)
            }
            layout::Univariant { .. } | layout::UntaggedUnion { .. } => C_u8(bcx.ccx, 0),
            layout::RawNullablePointer { nndiscr, .. } => {
                let cmp = if nndiscr == 0 { llvm::IntEQ } else { llvm::IntNE };
                let discr = bcx.load(self.llval, self.alignment.to_align());
                bcx.icmp(cmp, discr, C_null(val_ty(discr)))
            }
            layout::StructWrappedNullablePointer { nndiscr, ref discrfield, .. } => {
                let llptrptr = self.gepi_struct_llfields_path(bcx, discrfield);
                let llptr = bcx.load(llptrptr, self.alignment.to_align());
                let cmp = if nndiscr == 0 { llvm::IntEQ } else { llvm::IntNE };
                bcx.icmp(cmp, llptr, C_null(val_ty(llptr)))
            },
            _ => bug!("{} is not an enum", l.ty)
        };
        let cast_to = type_of::immediate_type_of(bcx.ccx, cast_to);
        bcx.intcast(val, cast_to, adt::is_discr_signed(&l))
    }

    /// Set the discriminant for a new value of the given case of the given
    /// representation.
    pub fn trans_set_discr(&self, bcx: &Builder<'a, 'tcx>, variant_index: usize) {
        let l = bcx.ccx.layout_of(self.ty.to_ty(bcx.tcx()));
        let to = l.ty.ty_adt_def().unwrap()
            .discriminant_for_variant(bcx.tcx(), variant_index)
            .to_u128_unchecked() as u64;
        match *l {
            layout::CEnum { discr, min, max, .. } => {
                adt::assert_discr_in_range(min, max, to);
                bcx.store(C_int(Type::from_integer(bcx.ccx, discr), to as i64),
                    self.llval, self.alignment.to_align());
            }
            layout::General { discr, .. } => {
                bcx.store(C_int(Type::from_integer(bcx.ccx, discr), to as i64),
                    bcx.struct_gep(self.llval, 0), self.alignment.to_align());
            }
            layout::Univariant { .. }
            | layout::UntaggedUnion { .. }
            | layout::Vector { .. } => {
                assert_eq!(to, 0);
            }
            layout::RawNullablePointer { nndiscr, .. } => {
                if to != nndiscr {
                    let llptrty = val_ty(self.llval).element_type();
                    bcx.store(C_null(llptrty), self.llval, self.alignment.to_align());
                }
            }
            layout::StructWrappedNullablePointer { nndiscr, ref discrfield, ref nonnull, .. } => {
                if to != nndiscr {
                    if target_sets_discr_via_memset(bcx) {
                        // Issue #34427: As workaround for LLVM bug on
                        // ARM, use memset of 0 on whole struct rather
                        // than storing null to single target field.
                        let llptr = bcx.pointercast(self.llval, Type::i8(bcx.ccx).ptr_to());
                        let fill_byte = C_u8(bcx.ccx, 0);
                        let size = C_usize(bcx.ccx, nonnull.stride().bytes());
                        let align = C_i32(bcx.ccx, nonnull.align.abi() as i32);
                        base::call_memset(bcx, llptr, fill_byte, size, align, false);
                    } else {
                        let llptrptr = self.gepi_struct_llfields_path(bcx, discrfield);
                        let llptrty = val_ty(llptrptr).element_type();
                        bcx.store(C_null(llptrty), llptrptr, self.alignment.to_align());
                    }
                }
            }
            _ => bug!("Cannot handle {} represented as {:#?}", l.ty, l)
        }
    }

    pub fn project_index(&self, bcx: &Builder<'a, 'tcx>, llindex: ValueRef) -> ValueRef {
        if let ty::TySlice(_) = self.ty.to_ty(bcx.tcx()).sty {
            // Slices already point to the array element type.
            bcx.inbounds_gep(self.llval, &[llindex])
        } else {
            let zero = common::C_usize(bcx.ccx, 0);
            bcx.inbounds_gep(self.llval, &[zero, llindex])
        }
    }

    pub fn storage_live(&self, bcx: &Builder<'a, 'tcx>) {
        bcx.lifetime_start(self.llval, bcx.ccx.size_of(self.ty.to_ty(bcx.tcx())));
    }

    pub fn storage_dead(&self, bcx: &Builder<'a, 'tcx>) {
        bcx.lifetime_end(self.llval, bcx.ccx.size_of(self.ty.to_ty(bcx.tcx())));
    }
}

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn trans_lvalue(&mut self,
                        bcx: &Builder<'a, 'tcx>,
                        lvalue: &mir::Lvalue<'tcx>)
                        -> LvalueRef<'tcx> {
        debug!("trans_lvalue(lvalue={:?})", lvalue);

        let ccx = bcx.ccx;
        let tcx = ccx.tcx();

        if let mir::Lvalue::Local(index) = *lvalue {
            match self.locals[index] {
                LocalRef::Lvalue(lvalue) => {
                    return lvalue;
                }
                LocalRef::Operand(..) => {
                    bug!("using operand local {:?} as lvalue", lvalue);
                }
            }
        }

        let result = match *lvalue {
            mir::Lvalue::Local(_) => bug!(), // handled above
            mir::Lvalue::Static(box mir::Static { def_id, ty }) => {
                LvalueRef::new_sized(consts::get_static(ccx, def_id),
                                     LvalueTy::from_ty(self.monomorphize(&ty)),
                                     Alignment::AbiAligned)
            },
            mir::Lvalue::Projection(box mir::Projection {
                ref base,
                elem: mir::ProjectionElem::Deref
            }) => {
                // Load the pointer from its location.
                self.trans_consume(bcx, base).deref()
            }
            mir::Lvalue::Projection(ref projection) => {
                let tr_base = self.trans_lvalue(bcx, &projection.base);
                let projected_ty = tr_base.ty.projection_ty(tcx, &projection.elem);
                let projected_ty = self.monomorphize(&projected_ty);
                let align = tr_base.alignment;

                let ((llprojected, align), llextra) = match projection.elem {
                    mir::ProjectionElem::Deref => bug!(),
                    mir::ProjectionElem::Field(ref field, _) => {
                        let has_metadata = self.ccx.shared()
                            .type_has_metadata(projected_ty.to_ty(tcx));
                        let llextra = if !has_metadata {
                            ptr::null_mut()
                        } else {
                            tr_base.llextra
                        };
                        (tr_base.trans_field_ptr(bcx, field.index()), llextra)
                    }
                    mir::ProjectionElem::Index(index) => {
                        let index = &mir::Operand::Consume(mir::Lvalue::Local(index));
                        let index = self.trans_operand(bcx, index);
                        let llindex = index.immediate();
                        ((tr_base.project_index(bcx, llindex), align), ptr::null_mut())
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: false,
                                                         min_length: _ } => {
                        let lloffset = C_usize(bcx.ccx, offset as u64);
                        ((tr_base.project_index(bcx, lloffset), align), ptr::null_mut())
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: true,
                                                         min_length: _ } => {
                        let lloffset = C_usize(bcx.ccx, offset as u64);
                        let lllen = tr_base.len(bcx.ccx);
                        let llindex = bcx.sub(lllen, lloffset);
                        ((tr_base.project_index(bcx, llindex), align), ptr::null_mut())
                    }
                    mir::ProjectionElem::Subslice { from, to } => {
                        let llbase = tr_base.project_index(bcx, C_usize(bcx.ccx, from as u64));

                        let base_ty = tr_base.ty.to_ty(bcx.tcx());
                        match base_ty.sty {
                            ty::TyArray(..) => {
                                // must cast the lvalue pointer type to the new
                                // array type (*[%_; new_len]).
                                let base_ty = self.monomorphized_lvalue_ty(lvalue);
                                let llbasety = type_of::type_of(bcx.ccx, base_ty).ptr_to();
                                let llbase = bcx.pointercast(llbase, llbasety);
                                ((llbase, align), ptr::null_mut())
                            }
                            ty::TySlice(..) => {
                                assert!(tr_base.llextra != ptr::null_mut());
                                let lllen = bcx.sub(tr_base.llextra,
                                                    C_usize(bcx.ccx, (from as u64)+(to as u64)));
                                ((llbase, align), lllen)
                            }
                            _ => bug!("unexpected type {:?} in Subslice", base_ty)
                        }
                    }
                    mir::ProjectionElem::Downcast(..) => {
                        ((tr_base.llval, align), tr_base.llextra)
                    }
                };
                LvalueRef {
                    llval: llprojected,
                    llextra,
                    ty: projected_ty,
                    alignment: align,
                }
            }
        };
        debug!("trans_lvalue(lvalue={:?}) => {:?}", lvalue, result);
        result
    }

    pub fn monomorphized_lvalue_ty(&self, lvalue: &mir::Lvalue<'tcx>) -> Ty<'tcx> {
        let tcx = self.ccx.tcx();
        let lvalue_ty = lvalue.ty(self.mir, tcx);
        self.monomorphize(&lvalue_ty.to_ty(tcx))
    }
}

