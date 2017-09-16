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
use rustc::ty::layout::{self, Align, Layout, LayoutOf};
use rustc::mir;
use rustc::mir::tcx::LvalueTy;
use rustc_data_structures::indexed_vec::Idx;
use abi;
use adt;
use base;
use builder::Builder;
use common::{self, CrateContext, C_usize, C_u8, C_u32, C_uint, C_int, C_null, val_ty};
use consts;
use type_of::LayoutLlvmExt;
use type_::Type;
use value::Value;
use glue;

use std::ptr;
use std::ops;

use super::{MirContext, LocalRef};
use super::operand::{OperandRef, OperandValue};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Alignment {
    Packed(Align),
    AbiAligned,
}

impl ops::BitOr for Alignment {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Alignment::Packed(a), Alignment::Packed(b)) => {
                Alignment::Packed(a.min(b))
            }
            (Alignment::Packed(x), _) | (_, Alignment::Packed(x)) => {
                Alignment::Packed(x)
            }
            (Alignment::AbiAligned, Alignment::AbiAligned) => {
                Alignment::AbiAligned
            }
        }
    }
}

impl<'a> From<&'a Layout> for Alignment {
    fn from(layout: &Layout) -> Self {
        let (packed, align) = match *layout {
            Layout::UntaggedUnion(ref un) => (un.packed, un.align),
            Layout::Univariant(ref variant) => (variant.packed, variant.align),
            _ => return Alignment::AbiAligned
        };
        if packed {
            Alignment::Packed(align)
        } else {
            Alignment::AbiAligned
        }
    }
}

impl Alignment {
    pub fn non_abi(self) -> Option<Align> {
        match self {
            Alignment::Packed(x) => Some(x),
            Alignment::AbiAligned => None,
        }
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
    pub fn new_sized(llval: ValueRef, ty: Ty<'tcx>, alignment: Alignment) -> LvalueRef<'tcx> {
        LvalueRef { llval, llextra: ptr::null_mut(), ty: LvalueTy::from_ty(ty), alignment }
    }

    pub fn alloca(bcx: &Builder<'a, 'tcx>, ty: Ty<'tcx>, name: &str) -> LvalueRef<'tcx> {
        debug!("alloca({:?}: {:?})", name, ty);
        let tmp = bcx.alloca(
            bcx.ccx.llvm_type_of(ty), name, bcx.ccx.over_align_of(ty));
        assert!(!ty.has_param_types());
        Self::new_sized(tmp, ty, Alignment::AbiAligned)
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

    pub fn load(&self, bcx: &Builder<'a, 'tcx>) -> OperandRef<'tcx> {
        debug!("LvalueRef::load: {:?}", self);

        assert!(!self.has_extra());

        let ty = self.ty.to_ty(bcx.tcx());

        if common::type_is_zero_size(bcx.ccx, ty) {
            return OperandRef::new_zst(bcx.ccx, ty);
        }

        let val = if common::type_is_fat_ptr(bcx.ccx, ty) {
            let data = self.project_field(bcx, abi::FAT_PTR_ADDR);
            let lldata = if ty.is_region_ptr() || ty.is_box() {
                bcx.load_nonnull(data.llval, data.alignment.non_abi())
            } else {
                bcx.load(data.llval, data.alignment.non_abi())
            };

            let extra = self.project_field(bcx, abi::FAT_PTR_EXTRA);
            let meta_ty = val_ty(extra.llval);
            // If the 'extra' field is a pointer, it's a vtable, so use load_nonnull
            // instead
            let llextra = if meta_ty.element_type().kind() == llvm::TypeKind::Pointer {
                bcx.load_nonnull(extra.llval, extra.alignment.non_abi())
            } else {
                bcx.load(extra.llval, extra.alignment.non_abi())
            };

            OperandValue::Pair(lldata, llextra)
        } else if common::type_is_imm_pair(bcx.ccx, ty) {
            OperandValue::Pair(
                self.project_field(bcx, 0).load(bcx).pack_if_pair(bcx).immediate(),
                self.project_field(bcx, 1).load(bcx).pack_if_pair(bcx).immediate())
        } else if common::type_is_immediate(bcx.ccx, ty) {
            let mut const_llval = ptr::null_mut();
            unsafe {
                let global = llvm::LLVMIsAGlobalVariable(self.llval);
                if !global.is_null() && llvm::LLVMIsGlobalConstant(global) == llvm::True {
                    const_llval = llvm::LLVMGetInitializer(global);
                }
            }

            let llval = if !const_llval.is_null() {
                const_llval
            } else if ty.is_bool() {
                bcx.load_range_assert(self.llval, 0, 2, llvm::False,
                    self.alignment.non_abi())
            } else if ty.is_char() {
                // a char is a Unicode codepoint, and so takes values from 0
                // to 0x10FFFF inclusive only.
                bcx.load_range_assert(self.llval, 0, 0x10FFFF + 1, llvm::False,
                    self.alignment.non_abi())
            } else if ty.is_region_ptr() || ty.is_box() || ty.is_fn() {
                bcx.load_nonnull(self.llval, self.alignment.non_abi())
            } else {
                bcx.load(self.llval, self.alignment.non_abi())
            };
            OperandValue::Immediate(base::to_immediate(bcx, llval, ty))
        } else {
            OperandValue::Ref(self.llval, self.alignment)
        };

        OperandRef { val, ty }
    }

    /// Access a field, at a point when the value's case is known.
    pub fn project_field(self, bcx: &Builder<'a, 'tcx>, ix: usize) -> LvalueRef<'tcx> {
        let ccx = bcx.ccx;
        let mut l = ccx.layout_of(self.ty.to_ty(bcx.tcx()));
        match self.ty {
            LvalueTy::Ty { .. } => {}
            LvalueTy::Downcast { variant_index, .. } => {
                l = l.for_variant(variant_index)
            }
        }
        let field = l.field(ccx, ix);
        let offset = l.fields.offset(ix).bytes();

        let alignment = self.alignment | Alignment::from(&*l);

        // Unions and newtypes only use an offset of 0.
        match *l {
            // FIXME(eddyb) The fields of a fat pointer aren't correct, especially
            // to unsized structs, we can't represent their pointee types in `Ty`.
            Layout::FatPointer { .. } => {}

            _ if offset == 0 => {
                let ty = ccx.llvm_type_of(field.ty);
                return LvalueRef {
                    llval: bcx.pointercast(self.llval, ty.ptr_to()),
                    llextra: if field.is_unsized() {
                        self.llextra
                    } else {
                        ptr::null_mut()
                    },
                    ty: LvalueTy::from_ty(field.ty),
                    alignment,
                };
            }

            _ => {}
        }

        // Discriminant field of enums.
        match *l {
            layout::NullablePointer { .. } if l.variant_index.is_none() => {
                let ty = ccx.llvm_type_of(field.ty);
                let size = field.size(ccx).bytes();

                // If the discriminant is not on a multiple of the primitive's size,
                // we need to go through i8*. Also assume the worst alignment.
                if offset % size != 0 {
                    let byte_ptr = bcx.pointercast(self.llval, Type::i8p(ccx));
                    let byte_ptr = bcx.inbounds_gep(byte_ptr, &[C_usize(ccx, offset)]);
                    let byte_align = Alignment::Packed(Align::from_bytes(1, 1).unwrap());
                    return LvalueRef::new_sized(
                        bcx.pointercast(byte_ptr, ty.ptr_to()), field.ty, byte_align);
                }

                let discr_ptr = bcx.pointercast(self.llval, ty.ptr_to());
                return LvalueRef::new_sized(
                    bcx.inbounds_gep(discr_ptr, &[C_usize(ccx, offset / size)]),
                    field.ty, alignment);
            }
            _ => {}
        }

        let simple = || {
            LvalueRef {
                llval: bcx.struct_gep(self.llval, l.llvm_field_index(ix)),
                llextra: if ccx.shared().type_has_metadata(field.ty) {
                    self.llextra
                } else {
                    ptr::null_mut()
                },
                ty: LvalueTy::from_ty(field.ty),
                alignment,
            }
        };

        // Check whether the variant being used is packed, if applicable.
        let is_packed = match (&*l, l.variant_index) {
            (&layout::Univariant(ref variant), _) => variant.packed,
            (&layout::NullablePointer { ref nonnull, .. }, _) => nonnull.packed,
            (&layout::General { ref variants, .. }, Some(v)) => variants[v].packed,
            _ => return simple()
        };

        // Simple case - we can just GEP the field
        //   * Packed struct - There is no alignment padding
        //   * Field is sized - pointer is properly aligned already
        if is_packed || !field.is_unsized() {
            return simple();
        }

        // If the type of the last field is [T], str or a foreign type, then we don't need to do
        // any adjusments
        match field.ty.sty {
            ty::TySlice(..) | ty::TyStr | ty::TyForeign(..) => return simple(),
            _ => ()
        }

        // There's no metadata available, log the case and just do the GEP.
        if !self.has_extra() {
            debug!("Unsized field `{}`, of `{:?}` has no metadata for adjustment",
                ix, Value(self.llval));
            return simple();
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

        let unaligned_offset = C_usize(ccx, offset);

        // Get the alignment of the field
        let (_, align) = glue::size_and_align_of_dst(bcx, field.ty, meta);

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
        let byte_ptr = bcx.pointercast(self.llval, Type::i8p(ccx));
        let byte_ptr = bcx.gep(byte_ptr, &[offset]);

        // Finally, cast back to the type expected
        let ll_fty = ccx.llvm_type_of(field.ty);
        debug!("struct_field_ptr: Field type is {:?}", ll_fty);

        LvalueRef {
            llval: bcx.pointercast(byte_ptr, ll_fty.ptr_to()),
            llextra: self.llextra,
            ty: LvalueTy::from_ty(field.ty),
            alignment,
        }
    }

    /// Obtain the actual discriminant of a value.
    pub fn trans_get_discr(self, bcx: &Builder<'a, 'tcx>, cast_to: Ty<'tcx>) -> ValueRef {
        let l = bcx.ccx.layout_of(self.ty.to_ty(bcx.tcx()));

        let cast_to = bcx.ccx.immediate_llvm_type_of(cast_to);
        match *l {
            layout::Univariant { .. } |
            layout::UntaggedUnion { .. } => return C_uint(cast_to, 0),
            _ => {}
        }

        let discr = self.project_field(bcx, 0);
        let discr_layout = bcx.ccx.layout_of(discr.ty.to_ty(bcx.tcx()));
        let discr_scalar = match discr_layout.abi {
            layout::Abi::Scalar(discr) => discr,
            _ => bug!("discriminant not scalar: {:#?}", discr_layout)
        };
        let (min, max) = match *l {
            layout::General { ref discr_range, .. } => (discr_range.start, discr_range.end),
            _ => (0, u64::max_value()),
        };
        let max_next = max.wrapping_add(1);
        let bits = discr_scalar.size(bcx.ccx).bits();
        assert!(bits <= 64);
        let mask = !0u64 >> (64 - bits);
        let lldiscr = match discr_scalar {
            // For a (max) discr of -1, max will be `-1 as usize`, which overflows.
            // However, that is fine here (it would still represent the full range),
            layout::Int(..) if max_next & mask != min & mask => {
                // llvm::ConstantRange can deal with ranges that wrap around,
                // so an overflow on (max + 1) is fine.
                bcx.load_range_assert(discr.llval, min, max_next,
                                      /* signed: */ llvm::True,
                                      discr.alignment.non_abi())
            }
            _ => {
                // i.e., if the range is everything.  The lo==hi case would be
                // rejected by the LLVM verifier (it would mean either an
                // empty set, which is impossible, or the entire range of the
                // type, which is pointless).
                bcx.load(discr.llval, discr.alignment.non_abi())
            }
        };
        match *l {
            layout::General { .. } => {
                let signed = match discr_scalar {
                    layout::Int(_, signed) => signed,
                    _ => false
                };
                bcx.intcast(lldiscr, cast_to, signed)
            }
            layout::NullablePointer { nndiscr, .. } => {
                let cmp = if nndiscr == 0 { llvm::IntEQ } else { llvm::IntNE };
                let zero = C_null(bcx.ccx.llvm_type_of(discr_layout.ty));
                bcx.intcast(bcx.icmp(cmp, lldiscr, zero), cast_to, false)
            }
            _ => bug!("{} is not an enum", l.ty)
        }
    }

    /// Set the discriminant for a new value of the given case of the given
    /// representation.
    pub fn trans_set_discr(&self, bcx: &Builder<'a, 'tcx>, variant_index: usize) {
        let l = bcx.ccx.layout_of(self.ty.to_ty(bcx.tcx()));
        let to = l.ty.ty_adt_def().unwrap()
            .discriminant_for_variant(bcx.tcx(), variant_index)
            .to_u128_unchecked() as u64;
        match *l {
            layout::General { .. } => {
                let ptr = self.project_field(bcx, 0);
                bcx.store(C_int(bcx.ccx.llvm_type_of(ptr.ty.to_ty(bcx.tcx())), to as i64),
                    ptr.llval, ptr.alignment.non_abi());
            }
            layout::NullablePointer { nndiscr, .. } => {
                if to != nndiscr {
                    let use_memset = match l.abi {
                        layout::Abi::Scalar(_) => false,
                        _ => target_sets_discr_via_memset(bcx)
                    };
                    if use_memset {
                        // Issue #34427: As workaround for LLVM bug on
                        // ARM, use memset of 0 on whole struct rather
                        // than storing null to single target field.
                        let llptr = bcx.pointercast(self.llval, Type::i8(bcx.ccx).ptr_to());
                        let fill_byte = C_u8(bcx.ccx, 0);
                        let (size, align) = l.size_and_align(bcx.ccx);
                        let size = C_usize(bcx.ccx, size.bytes());
                        let align = C_u32(bcx.ccx, align.abi() as u32);
                        base::call_memset(bcx, llptr, fill_byte, size, align, false);
                    } else {
                        let ptr = self.project_field(bcx, 0);
                        bcx.store(C_null(bcx.ccx.llvm_type_of(ptr.ty.to_ty(bcx.tcx()))),
                            ptr.llval, ptr.alignment.non_abi());
                    }
                }
            }
            _ => {
                assert_eq!(to, 0);
            }
        }
    }

    pub fn project_index(&self, bcx: &Builder<'a, 'tcx>, llindex: ValueRef)
                         -> LvalueRef<'tcx> {
        let ptr = bcx.inbounds_gep(self.llval, &[common::C_usize(bcx.ccx, 0), llindex]);
        let elem_ty = self.ty.to_ty(bcx.tcx()).builtin_index().unwrap();
        LvalueRef::new_sized(ptr, elem_ty, self.alignment)
    }

    pub fn project_downcast(&self, bcx: &Builder<'a, 'tcx>, variant_index: usize)
                            -> LvalueRef<'tcx> {
        let ty = self.ty.to_ty(bcx.tcx());
        if let ty::TyAdt(adt_def, substs) = ty.sty {
            let mut downcast = *self;
            downcast.ty = LvalueTy::Downcast {
                adt_def,
                substs,
                variant_index,
            };

            // If this is an enum, cast to the appropriate variant struct type.
            let layout = bcx.ccx.layout_of(ty).for_variant(variant_index);
            if let layout::General { ref variants, .. } = *layout {
                let st = &variants[variant_index];
                let variant_ty = Type::struct_(bcx.ccx,
                    &adt::struct_llfields(bcx.ccx, layout, st), st.packed);
                downcast.llval = bcx.pointercast(downcast.llval, variant_ty.ptr_to());
            }

            downcast
        } else {
            bug!("unexpected type `{}` in LvalueRef::project_downcast", ty)
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
                                     self.monomorphize(&ty),
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

                match projection.elem {
                    mir::ProjectionElem::Deref => bug!(),
                    mir::ProjectionElem::Field(ref field, _) => {
                        tr_base.project_field(bcx, field.index())
                    }
                    mir::ProjectionElem::Index(index) => {
                        let index = &mir::Operand::Consume(mir::Lvalue::Local(index));
                        let index = self.trans_operand(bcx, index);
                        let llindex = index.immediate();
                        tr_base.project_index(bcx, llindex)
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: false,
                                                         min_length: _ } => {
                        let lloffset = C_usize(bcx.ccx, offset as u64);
                        tr_base.project_index(bcx, lloffset)
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: true,
                                                         min_length: _ } => {
                        let lloffset = C_usize(bcx.ccx, offset as u64);
                        let lllen = tr_base.len(bcx.ccx);
                        let llindex = bcx.sub(lllen, lloffset);
                        tr_base.project_index(bcx, llindex)
                    }
                    mir::ProjectionElem::Subslice { from, to } => {
                        let mut subslice = tr_base.project_index(bcx,
                            C_usize(bcx.ccx, from as u64));
                        subslice.ty = tr_base.ty.projection_ty(tcx, &projection.elem);
                        subslice.ty = self.monomorphize(&subslice.ty);

                        match subslice.ty.to_ty(tcx).sty {
                            ty::TyArray(..) => {}
                            ty::TySlice(..) => {
                                assert!(tr_base.has_extra());
                                subslice.llextra = bcx.sub(tr_base.llextra,
                                    C_usize(bcx.ccx, (from as u64) + (to as u64)));
                            }
                            _ => bug!("unexpected type {:?} in Subslice", subslice.ty)
                        }

                        // Cast the lvalue pointer type to the new
                        // array or slice type (*[%_; new_len]).
                        subslice.llval = bcx.pointercast(subslice.llval,
                            bcx.ccx.llvm_type_of(subslice.ty.to_ty(tcx)).ptr_to());

                        subslice
                    }
                    mir::ProjectionElem::Downcast(_, v) => {
                        tr_base.project_downcast(bcx, v)
                    }
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

