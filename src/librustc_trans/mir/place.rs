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
use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, Align, TyLayout, LayoutOf};
use rustc::mir;
use rustc::mir::tcx::PlaceTy;
use rustc_data_structures::indexed_vec::Idx;
use base;
use builder::Builder;
use common::{CrateContext, C_usize, C_u8, C_u32, C_uint, C_int, C_null, C_uint_big};
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

impl<'a> From<TyLayout<'a>> for Alignment {
    fn from(layout: TyLayout) -> Self {
        if layout.is_packed() {
            Alignment::Packed(layout.align)
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

#[derive(Copy, Clone, Debug)]
pub struct PlaceRef<'tcx> {
    /// Pointer to the contents of the place
    pub llval: ValueRef,

    /// This place's extra data if it is unsized, or null
    pub llextra: ValueRef,

    /// Monomorphized type of this place, including variant information
    pub layout: TyLayout<'tcx>,

    /// Whether this place is known to be aligned according to its layout
    pub alignment: Alignment,
}

impl<'a, 'tcx> PlaceRef<'tcx> {
    pub fn new_sized(llval: ValueRef,
                     layout: TyLayout<'tcx>,
                     alignment: Alignment)
                     -> PlaceRef<'tcx> {
        PlaceRef {
            llval,
            llextra: ptr::null_mut(),
            layout,
            alignment
        }
    }

    pub fn alloca(bcx: &Builder<'a, 'tcx>, layout: TyLayout<'tcx>, name: &str)
                  -> PlaceRef<'tcx> {
        debug!("alloca({:?}: {:?})", name, layout);
        let tmp = bcx.alloca(layout.llvm_type(bcx.ccx), name, layout.align);
        Self::new_sized(tmp, layout, Alignment::AbiAligned)
    }

    pub fn len(&self, ccx: &CrateContext<'a, 'tcx>) -> ValueRef {
        if let layout::FieldPlacement::Array { count, .. } = self.layout.fields {
            if self.layout.is_unsized() {
                assert!(self.has_extra());
                assert_eq!(count, 0);
                self.llextra
            } else {
                C_usize(ccx, count)
            }
        } else {
            bug!("unexpected layout `{:#?}` in PlaceRef::len", self.layout)
        }
    }

    pub fn has_extra(&self) -> bool {
        !self.llextra.is_null()
    }

    pub fn load(&self, bcx: &Builder<'a, 'tcx>) -> OperandRef<'tcx> {
        debug!("PlaceRef::load: {:?}", self);

        assert!(!self.has_extra());

        if self.layout.is_zst() {
            return OperandRef::new_zst(bcx.ccx, self.layout);
        }

        let scalar_load_metadata = |load, scalar: &layout::Scalar| {
            let (min, max) = (scalar.valid_range.start, scalar.valid_range.end);
            let max_next = max.wrapping_add(1);
            let bits = scalar.value.size(bcx.ccx).bits();
            assert!(bits <= 128);
            let mask = !0u128 >> (128 - bits);
            // For a (max) value of -1, max will be `-1 as usize`, which overflows.
            // However, that is fine here (it would still represent the full range),
            // i.e., if the range is everything.  The lo==hi case would be
            // rejected by the LLVM verifier (it would mean either an
            // empty set, which is impossible, or the entire range of the
            // type, which is pointless).
            match scalar.value {
                layout::Int(..) if max_next & mask != min & mask => {
                    // llvm::ConstantRange can deal with ranges that wrap around,
                    // so an overflow on (max + 1) is fine.
                    bcx.range_metadata(load, min..max_next);
                }
                layout::Pointer if 0 < min && min < max => {
                    bcx.nonnull_metadata(load);
                }
                _ => {}
            }
        };

        let val = if self.layout.is_llvm_immediate() {
            let mut const_llval = ptr::null_mut();
            unsafe {
                let global = llvm::LLVMIsAGlobalVariable(self.llval);
                if !global.is_null() && llvm::LLVMIsGlobalConstant(global) == llvm::True {
                    const_llval = llvm::LLVMGetInitializer(global);
                }
            }

            let llval = if !const_llval.is_null() {
                const_llval
            } else {
                let load = bcx.load(self.llval, self.alignment.non_abi());
                if let layout::Abi::Scalar(ref scalar) = self.layout.abi {
                    scalar_load_metadata(load, scalar);
                }
                load
            };
            OperandValue::Immediate(base::to_immediate(bcx, llval, self.layout))
        } else if let layout::Abi::ScalarPair(ref a, ref b) = self.layout.abi {
            let load = |i, scalar: &layout::Scalar| {
                let mut llptr = bcx.struct_gep(self.llval, i as u64);
                // Make sure to always load i1 as i8.
                if scalar.is_bool() {
                    llptr = bcx.pointercast(llptr, Type::i8p(bcx.ccx));
                }
                let load = bcx.load(llptr, self.alignment.non_abi());
                scalar_load_metadata(load, scalar);
                if scalar.is_bool() {
                    bcx.trunc(load, Type::i1(bcx.ccx))
                } else {
                    load
                }
            };
            OperandValue::Pair(load(0, a), load(1, b))
        } else {
            OperandValue::Ref(self.llval, self.alignment)
        };

        OperandRef { val, layout: self.layout }
    }

    /// Access a field, at a point when the value's case is known.
    pub fn project_field(self, bcx: &Builder<'a, 'tcx>, ix: usize) -> PlaceRef<'tcx> {
        let ccx = bcx.ccx;
        let field = self.layout.field(ccx, ix);
        let offset = self.layout.fields.offset(ix);
        let alignment = self.alignment | Alignment::from(self.layout);

        let simple = || {
            // Unions and newtypes only use an offset of 0.
            let llval = if offset.bytes() == 0 {
                self.llval
            } else if let layout::Abi::ScalarPair(ref a, ref b) = self.layout.abi {
                // Offsets have to match either first or second field.
                assert_eq!(offset, a.value.size(ccx).abi_align(b.value.align(ccx)));
                bcx.struct_gep(self.llval, 1)
            } else {
                bcx.struct_gep(self.llval, self.layout.llvm_field_index(ix))
            };
            PlaceRef {
                // HACK(eddyb) have to bitcast pointers until LLVM removes pointee types.
                llval: bcx.pointercast(llval, field.llvm_type(ccx).ptr_to()),
                llextra: if ccx.shared().type_has_metadata(field.ty) {
                    self.llextra
                } else {
                    ptr::null_mut()
                },
                layout: field,
                alignment,
            }
        };

        // Simple case - we can just GEP the field
        //   * Packed struct - There is no alignment padding
        //   * Field is sized - pointer is properly aligned already
        if self.layout.is_packed() || !field.is_unsized() {
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

        let unaligned_offset = C_usize(ccx, offset.bytes());

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
        let ll_fty = field.llvm_type(ccx);
        debug!("struct_field_ptr: Field type is {:?}", ll_fty);

        PlaceRef {
            llval: bcx.pointercast(byte_ptr, ll_fty.ptr_to()),
            llextra: self.llextra,
            layout: field,
            alignment,
        }
    }

    /// Obtain the actual discriminant of a value.
    pub fn trans_get_discr(self, bcx: &Builder<'a, 'tcx>, cast_to: Ty<'tcx>) -> ValueRef {
        let cast_to = bcx.ccx.layout_of(cast_to).immediate_llvm_type(bcx.ccx);
        match self.layout.variants {
            layout::Variants::Single { index } => {
                return C_uint(cast_to, index as u64);
            }
            layout::Variants::Tagged { .. } |
            layout::Variants::NicheFilling { .. } => {},
        }

        let discr = self.project_field(bcx, 0);
        let lldiscr = discr.load(bcx).immediate();
        match self.layout.variants {
            layout::Variants::Single { .. } => bug!(),
            layout::Variants::Tagged { ref discr, .. } => {
                let signed = match discr.value {
                    layout::Int(_, signed) => signed,
                    _ => false
                };
                bcx.intcast(lldiscr, cast_to, signed)
            }
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                let niche_llty = discr.layout.immediate_llvm_type(bcx.ccx);
                if niche_variants.start == niche_variants.end {
                    // FIXME(eddyb) Check the actual primitive type here.
                    let niche_llval = if niche_start == 0 {
                        // HACK(eddyb) Using `C_null` as it works on all types.
                        C_null(niche_llty)
                    } else {
                        C_uint_big(niche_llty, niche_start)
                    };
                    bcx.select(bcx.icmp(llvm::IntEQ, lldiscr, niche_llval),
                        C_uint(cast_to, niche_variants.start as u64),
                        C_uint(cast_to, dataful_variant as u64))
                } else {
                    // Rebase from niche values to discriminant values.
                    let delta = niche_start.wrapping_sub(niche_variants.start as u128);
                    let lldiscr = bcx.sub(lldiscr, C_uint_big(niche_llty, delta));
                    let lldiscr_max = C_uint(niche_llty, niche_variants.end as u64);
                    bcx.select(bcx.icmp(llvm::IntULE, lldiscr, lldiscr_max),
                        bcx.intcast(lldiscr, cast_to, false),
                        C_uint(cast_to, dataful_variant as u64))
                }
            }
        }
    }

    /// Set the discriminant for a new value of the given case of the given
    /// representation.
    pub fn trans_set_discr(&self, bcx: &Builder<'a, 'tcx>, variant_index: usize) {
            if self.layout.for_variant(bcx.ccx, variant_index).abi == layout::Abi::Uninhabited {
                return;
            }
            match self.layout.variants {
            layout::Variants::Single { index } => {
                assert_eq!(index, variant_index);
            }
            layout::Variants::Tagged { .. } => {
                let ptr = self.project_field(bcx, 0);
                let to = self.layout.ty.ty_adt_def().unwrap()
                    .discriminant_for_variant(bcx.tcx(), variant_index)
                    .to_u128_unchecked() as u64;
                bcx.store(C_int(ptr.layout.llvm_type(bcx.ccx), to as i64),
                    ptr.llval, ptr.alignment.non_abi());
            }
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                if variant_index != dataful_variant {
                    if bcx.sess().target.target.arch == "arm" ||
                       bcx.sess().target.target.arch == "aarch64" {
                        // Issue #34427: As workaround for LLVM bug on ARM,
                        // use memset of 0 before assigning niche value.
                        let llptr = bcx.pointercast(self.llval, Type::i8(bcx.ccx).ptr_to());
                        let fill_byte = C_u8(bcx.ccx, 0);
                        let (size, align) = self.layout.size_and_align();
                        let size = C_usize(bcx.ccx, size.bytes());
                        let align = C_u32(bcx.ccx, align.abi() as u32);
                        base::call_memset(bcx, llptr, fill_byte, size, align, false);
                    }

                    let niche = self.project_field(bcx, 0);
                    let niche_llty = niche.layout.immediate_llvm_type(bcx.ccx);
                    let niche_value = ((variant_index - niche_variants.start) as u128)
                        .wrapping_add(niche_start);
                    // FIXME(eddyb) Check the actual primitive type here.
                    let niche_llval = if niche_value == 0 {
                        // HACK(eddyb) Using `C_null` as it works on all types.
                        C_null(niche_llty)
                    } else {
                        C_uint_big(niche_llty, niche_value)
                    };
                    OperandValue::Immediate(niche_llval).store(bcx, niche);
                }
            }
        }
    }

    pub fn project_index(&self, bcx: &Builder<'a, 'tcx>, llindex: ValueRef)
                         -> PlaceRef<'tcx> {
        PlaceRef {
            llval: bcx.inbounds_gep(self.llval, &[C_usize(bcx.ccx, 0), llindex]),
            llextra: ptr::null_mut(),
            layout: self.layout.field(bcx.ccx, 0),
            alignment: self.alignment
        }
    }

    pub fn project_downcast(&self, bcx: &Builder<'a, 'tcx>, variant_index: usize)
                            -> PlaceRef<'tcx> {
        let mut downcast = *self;
        downcast.layout = self.layout.for_variant(bcx.ccx, variant_index);

        // Cast to the appropriate variant struct type.
        let variant_ty = downcast.layout.llvm_type(bcx.ccx);
        downcast.llval = bcx.pointercast(downcast.llval, variant_ty.ptr_to());

        downcast
    }

    pub fn storage_live(&self, bcx: &Builder<'a, 'tcx>) {
        bcx.lifetime_start(self.llval, self.layout.size);
    }

    pub fn storage_dead(&self, bcx: &Builder<'a, 'tcx>) {
        bcx.lifetime_end(self.llval, self.layout.size);
    }
}

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn trans_place(&mut self,
                        bcx: &Builder<'a, 'tcx>,
                        place: &mir::Place<'tcx>)
                        -> PlaceRef<'tcx> {
        debug!("trans_place(place={:?})", place);

        let ccx = bcx.ccx;
        let tcx = ccx.tcx();

        if let mir::Place::Local(index) = *place {
            match self.locals[index] {
                LocalRef::Place(place) => {
                    return place;
                }
                LocalRef::Operand(..) => {
                    bug!("using operand local {:?} as place", place);
                }
            }
        }

        let result = match *place {
            mir::Place::Local(_) => bug!(), // handled above
            mir::Place::Static(box mir::Static { def_id, ty }) => {
                PlaceRef::new_sized(consts::get_static(ccx, def_id),
                                     ccx.layout_of(self.monomorphize(&ty)),
                                     Alignment::AbiAligned)
            },
            mir::Place::Projection(box mir::Projection {
                ref base,
                elem: mir::ProjectionElem::Deref
            }) => {
                // Load the pointer from its location.
                self.trans_consume(bcx, base).deref(bcx.ccx)
            }
            mir::Place::Projection(ref projection) => {
                let tr_base = self.trans_place(bcx, &projection.base);

                match projection.elem {
                    mir::ProjectionElem::Deref => bug!(),
                    mir::ProjectionElem::Field(ref field, _) => {
                        tr_base.project_field(bcx, field.index())
                    }
                    mir::ProjectionElem::Index(index) => {
                        let index = &mir::Operand::Copy(mir::Place::Local(index));
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
                        let projected_ty = PlaceTy::Ty { ty: tr_base.layout.ty }
                            .projection_ty(tcx, &projection.elem).to_ty(bcx.tcx());
                        subslice.layout = bcx.ccx.layout_of(self.monomorphize(&projected_ty));

                        if subslice.layout.is_unsized() {
                            assert!(tr_base.has_extra());
                            subslice.llextra = bcx.sub(tr_base.llextra,
                                C_usize(bcx.ccx, (from as u64) + (to as u64)));
                        }

                        // Cast the place pointer type to the new
                        // array or slice type (*[%_; new_len]).
                        subslice.llval = bcx.pointercast(subslice.llval,
                            subslice.layout.llvm_type(bcx.ccx).ptr_to());

                        subslice
                    }
                    mir::ProjectionElem::Downcast(_, v) => {
                        tr_base.project_downcast(bcx, v)
                    }
                }
            }
        };
        debug!("trans_place(place={:?}) => {:?}", place, result);
        result
    }

    pub fn monomorphized_place_ty(&self, place: &mir::Place<'tcx>) -> Ty<'tcx> {
        let tcx = self.ccx.tcx();
        let place_ty = place.ty(self.mir, tcx);
        self.monomorphize(&place_ty.to_ty(tcx))
    }
}

