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
use common::{CodegenCx, C_usize, C_u8, C_u32, C_uint, C_int, C_null, C_uint_big};
use consts;
use type_of::LayoutLlvmExt;
use type_::Type;
use value::Value;
use glue;

use std::ptr;

use super::{FunctionCx, LocalRef};
use super::operand::{OperandRef, OperandValue};

#[derive(Copy, Clone, Debug)]
pub struct PlaceRef<'tcx> {
    /// Pointer to the contents of the place
    pub llval: ValueRef,

    /// This place's extra data if it is unsized, or null
    pub llextra: ValueRef,

    /// Monomorphized type of this place, including variant information
    pub layout: TyLayout<'tcx>,

    /// What alignment we know for this place
    pub align: Align,
}

impl<'a, 'tcx> PlaceRef<'tcx> {
    pub fn new_sized(llval: ValueRef,
                     layout: TyLayout<'tcx>,
                     align: Align)
                     -> PlaceRef<'tcx> {
        PlaceRef {
            llval,
            llextra: ptr::null_mut(),
            layout,
            align
        }
    }

    pub fn alloca(bx: &Builder<'a, 'tcx>, layout: TyLayout<'tcx>, name: &str)
                  -> PlaceRef<'tcx> {
        debug!("alloca({:?}: {:?})", name, layout);
        let tmp = bx.alloca(layout.llvm_type(bx.cx), name, layout.align);
        Self::new_sized(tmp, layout, layout.align)
    }

    pub fn len(&self, cx: &CodegenCx<'a, 'tcx>) -> ValueRef {
        if let layout::FieldPlacement::Array { count, .. } = self.layout.fields {
            if self.layout.is_unsized() {
                assert!(self.has_extra());
                assert_eq!(count, 0);
                self.llextra
            } else {
                C_usize(cx, count)
            }
        } else {
            bug!("unexpected layout `{:#?}` in PlaceRef::len", self.layout)
        }
    }

    pub fn has_extra(&self) -> bool {
        !self.llextra.is_null()
    }

    pub fn load(&self, bx: &Builder<'a, 'tcx>) -> OperandRef<'tcx> {
        debug!("PlaceRef::load: {:?}", self);

        assert!(!self.has_extra());

        if self.layout.is_zst() {
            return OperandRef::new_zst(bx.cx, self.layout);
        }

        let scalar_load_metadata = |load, scalar: &layout::Scalar| {
            let (min, max) = (scalar.valid_range.start, scalar.valid_range.end);
            let max_next = max.wrapping_add(1);
            let bits = scalar.value.size(bx.cx).bits();
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
                    bx.range_metadata(load, min..max_next);
                }
                layout::Pointer if 0 < min && min < max => {
                    bx.nonnull_metadata(load);
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
                let load = bx.load(self.llval, self.align);
                if let layout::Abi::Scalar(ref scalar) = self.layout.abi {
                    scalar_load_metadata(load, scalar);
                }
                load
            };
            OperandValue::Immediate(base::to_immediate(bx, llval, self.layout))
        } else if let layout::Abi::ScalarPair(ref a, ref b) = self.layout.abi {
            let load = |i, scalar: &layout::Scalar| {
                let mut llptr = bx.struct_gep(self.llval, i as u64);
                // Make sure to always load i1 as i8.
                if scalar.is_bool() {
                    llptr = bx.pointercast(llptr, Type::i8p(bx.cx));
                }
                let load = bx.load(llptr, self.align);
                scalar_load_metadata(load, scalar);
                if scalar.is_bool() {
                    bx.trunc(load, Type::i1(bx.cx))
                } else {
                    load
                }
            };
            OperandValue::Pair(load(0, a), load(1, b))
        } else {
            OperandValue::Ref(self.llval, self.align)
        };

        OperandRef { val, layout: self.layout }
    }

    /// Access a field, at a point when the value's case is known.
    pub fn project_field(self, bx: &Builder<'a, 'tcx>, ix: usize) -> PlaceRef<'tcx> {
        let cx = bx.cx;
        let field = self.layout.field(cx, ix);
        let offset = self.layout.fields.offset(ix);
        let align = self.align.min(self.layout.align).min(field.align);

        let simple = || {
            // Unions and newtypes only use an offset of 0.
            let llval = if offset.bytes() == 0 {
                self.llval
            } else if let layout::Abi::ScalarPair(ref a, ref b) = self.layout.abi {
                // Offsets have to match either first or second field.
                assert_eq!(offset, a.value.size(cx).abi_align(b.value.align(cx)));
                bx.struct_gep(self.llval, 1)
            } else {
                bx.struct_gep(self.llval, self.layout.llvm_field_index(ix))
            };
            PlaceRef {
                // HACK(eddyb) have to bitcast pointers until LLVM removes pointee types.
                llval: bx.pointercast(llval, field.llvm_type(cx).ptr_to()),
                llextra: if cx.type_has_metadata(field.ty) {
                    self.llextra
                } else {
                    ptr::null_mut()
                },
                layout: field,
                align,
            }
        };

        // Simple cases, which don't need DST adjustment:
        //   * no metadata available - just log the case
        //   * known alignment - sized types, [T], str or a foreign type
        //   * packed struct - there is no alignment padding
        match field.ty.sty {
            _ if !self.has_extra() => {
                debug!("Unsized field `{}`, of `{:?}` has no metadata for adjustment",
                    ix, Value(self.llval));
                return simple();
            }
            _ if !field.is_unsized() => return simple(),
            ty::TySlice(..) | ty::TyStr | ty::TyForeign(..) => return simple(),
            ty::TyAdt(def, _) => {
                if def.repr.packed() {
                    // FIXME(eddyb) generalize the adjustment when we
                    // start supporting packing to larger alignments.
                    assert_eq!(self.layout.align.abi(), 1);
                    return simple();
                }
            }
            _ => {}
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

        let unaligned_offset = C_usize(cx, offset.bytes());

        // Get the alignment of the field
        let (_, unsized_align) = glue::size_and_align_of_dst(bx, field.ty, meta);

        // Bump the unaligned offset up to the appropriate alignment using the
        // following expression:
        //
        //   (unaligned offset + (align - 1)) & -align

        // Calculate offset
        let align_sub_1 = bx.sub(unsized_align, C_usize(cx, 1u64));
        let offset = bx.and(bx.add(unaligned_offset, align_sub_1),
        bx.neg(unsized_align));

        debug!("struct_field_ptr: DST field offset: {:?}", Value(offset));

        // Cast and adjust pointer
        let byte_ptr = bx.pointercast(self.llval, Type::i8p(cx));
        let byte_ptr = bx.gep(byte_ptr, &[offset]);

        // Finally, cast back to the type expected
        let ll_fty = field.llvm_type(cx);
        debug!("struct_field_ptr: Field type is {:?}", ll_fty);

        PlaceRef {
            llval: bx.pointercast(byte_ptr, ll_fty.ptr_to()),
            llextra: self.llextra,
            layout: field,
            align,
        }
    }

    /// Obtain the actual discriminant of a value.
    pub fn trans_get_discr(self, bx: &Builder<'a, 'tcx>, cast_to: Ty<'tcx>) -> ValueRef {
        let cast_to = bx.cx.layout_of(cast_to).immediate_llvm_type(bx.cx);
        match self.layout.variants {
            layout::Variants::Single { index } => {
                return C_uint(cast_to, index as u64);
            }
            layout::Variants::Tagged { .. } |
            layout::Variants::NicheFilling { .. } => {},
        }

        let discr = self.project_field(bx, 0);
        let lldiscr = discr.load(bx).immediate();
        match self.layout.variants {
            layout::Variants::Single { .. } => bug!(),
            layout::Variants::Tagged { ref discr, .. } => {
                let signed = match discr.value {
                    layout::Int(_, signed) => signed,
                    _ => false
                };
                bx.intcast(lldiscr, cast_to, signed)
            }
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                let niche_llty = discr.layout.immediate_llvm_type(bx.cx);
                if niche_variants.start == niche_variants.end {
                    // FIXME(eddyb) Check the actual primitive type here.
                    let niche_llval = if niche_start == 0 {
                        // HACK(eddyb) Using `C_null` as it works on all types.
                        C_null(niche_llty)
                    } else {
                        C_uint_big(niche_llty, niche_start)
                    };
                    bx.select(bx.icmp(llvm::IntEQ, lldiscr, niche_llval),
                        C_uint(cast_to, niche_variants.start as u64),
                        C_uint(cast_to, dataful_variant as u64))
                } else {
                    // Rebase from niche values to discriminant values.
                    let delta = niche_start.wrapping_sub(niche_variants.start as u128);
                    let lldiscr = bx.sub(lldiscr, C_uint_big(niche_llty, delta));
                    let lldiscr_max = C_uint(niche_llty, niche_variants.end as u64);
                    bx.select(bx.icmp(llvm::IntULE, lldiscr, lldiscr_max),
                        bx.intcast(lldiscr, cast_to, false),
                        C_uint(cast_to, dataful_variant as u64))
                }
            }
        }
    }

    /// Set the discriminant for a new value of the given case of the given
    /// representation.
    pub fn trans_set_discr(&self, bx: &Builder<'a, 'tcx>, variant_index: usize) {
        if self.layout.for_variant(bx.cx, variant_index).abi == layout::Abi::Uninhabited {
            return;
        }
        match self.layout.variants {
            layout::Variants::Single { index } => {
                assert_eq!(index, variant_index);
            }
            layout::Variants::Tagged { .. } => {
                let ptr = self.project_field(bx, 0);
                let to = self.layout.ty.ty_adt_def().unwrap()
                    .discriminant_for_variant(bx.tcx(), variant_index)
                    .val as u64;
                bx.store(C_int(ptr.layout.llvm_type(bx.cx), to as i64),
                    ptr.llval, ptr.align);
            }
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                if variant_index != dataful_variant {
                    if bx.sess().target.target.arch == "arm" ||
                       bx.sess().target.target.arch == "aarch64" {
                        // Issue #34427: As workaround for LLVM bug on ARM,
                        // use memset of 0 before assigning niche value.
                        let llptr = bx.pointercast(self.llval, Type::i8(bx.cx).ptr_to());
                        let fill_byte = C_u8(bx.cx, 0);
                        let (size, align) = self.layout.size_and_align();
                        let size = C_usize(bx.cx, size.bytes());
                        let align = C_u32(bx.cx, align.abi() as u32);
                        base::call_memset(bx, llptr, fill_byte, size, align, false);
                    }

                    let niche = self.project_field(bx, 0);
                    let niche_llty = niche.layout.immediate_llvm_type(bx.cx);
                    let niche_value = ((variant_index - niche_variants.start) as u128)
                        .wrapping_add(niche_start);
                    // FIXME(eddyb) Check the actual primitive type here.
                    let niche_llval = if niche_value == 0 {
                        // HACK(eddyb) Using `C_null` as it works on all types.
                        C_null(niche_llty)
                    } else {
                        C_uint_big(niche_llty, niche_value)
                    };
                    OperandValue::Immediate(niche_llval).store(bx, niche);
                }
            }
        }
    }

    pub fn project_index(&self, bx: &Builder<'a, 'tcx>, llindex: ValueRef)
                         -> PlaceRef<'tcx> {
        PlaceRef {
            llval: bx.inbounds_gep(self.llval, &[C_usize(bx.cx, 0), llindex]),
            llextra: ptr::null_mut(),
            layout: self.layout.field(bx.cx, 0),
            align: self.align
        }
    }

    pub fn project_downcast(&self, bx: &Builder<'a, 'tcx>, variant_index: usize)
                            -> PlaceRef<'tcx> {
        let mut downcast = *self;
        downcast.layout = self.layout.for_variant(bx.cx, variant_index);

        // Cast to the appropriate variant struct type.
        let variant_ty = downcast.layout.llvm_type(bx.cx);
        downcast.llval = bx.pointercast(downcast.llval, variant_ty.ptr_to());

        downcast
    }

    pub fn storage_live(&self, bx: &Builder<'a, 'tcx>) {
        bx.lifetime_start(self.llval, self.layout.size);
    }

    pub fn storage_dead(&self, bx: &Builder<'a, 'tcx>) {
        bx.lifetime_end(self.llval, self.layout.size);
    }
}

impl<'a, 'tcx> FunctionCx<'a, 'tcx> {
    pub fn trans_place(&mut self,
                        bx: &Builder<'a, 'tcx>,
                        place: &mir::Place<'tcx>)
                        -> PlaceRef<'tcx> {
        debug!("trans_place(place={:?})", place);

        let cx = bx.cx;
        let tcx = cx.tcx;

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
                let layout = cx.layout_of(self.monomorphize(&ty));
                PlaceRef::new_sized(consts::get_static(cx, def_id), layout, layout.align)
            },
            mir::Place::Projection(box mir::Projection {
                ref base,
                elem: mir::ProjectionElem::Deref
            }) => {
                // Load the pointer from its location.
                self.trans_consume(bx, base).deref(bx.cx)
            }
            mir::Place::Projection(ref projection) => {
                let tr_base = self.trans_place(bx, &projection.base);

                match projection.elem {
                    mir::ProjectionElem::Deref => bug!(),
                    mir::ProjectionElem::Field(ref field, _) => {
                        tr_base.project_field(bx, field.index())
                    }
                    mir::ProjectionElem::Index(index) => {
                        let index = &mir::Operand::Copy(mir::Place::Local(index));
                        let index = self.trans_operand(bx, index);
                        let llindex = index.immediate();
                        tr_base.project_index(bx, llindex)
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: false,
                                                         min_length: _ } => {
                        let lloffset = C_usize(bx.cx, offset as u64);
                        tr_base.project_index(bx, lloffset)
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: true,
                                                         min_length: _ } => {
                        let lloffset = C_usize(bx.cx, offset as u64);
                        let lllen = tr_base.len(bx.cx);
                        let llindex = bx.sub(lllen, lloffset);
                        tr_base.project_index(bx, llindex)
                    }
                    mir::ProjectionElem::Subslice { from, to } => {
                        let mut subslice = tr_base.project_index(bx,
                            C_usize(bx.cx, from as u64));
                        let projected_ty = PlaceTy::Ty { ty: tr_base.layout.ty }
                            .projection_ty(tcx, &projection.elem).to_ty(bx.tcx());
                        subslice.layout = bx.cx.layout_of(self.monomorphize(&projected_ty));

                        if subslice.layout.is_unsized() {
                            assert!(tr_base.has_extra());
                            subslice.llextra = bx.sub(tr_base.llextra,
                                C_usize(bx.cx, (from as u64) + (to as u64)));
                        }

                        // Cast the place pointer type to the new
                        // array or slice type (*[%_; new_len]).
                        subslice.llval = bx.pointercast(subslice.llval,
                            subslice.layout.llvm_type(bx.cx).ptr_to());

                        subslice
                    }
                    mir::ProjectionElem::Downcast(_, v) => {
                        tr_base.project_downcast(bx, v)
                    }
                }
            }
        };
        debug!("trans_place(place={:?}) => {:?}", place, result);
        result
    }

    pub fn monomorphized_place_ty(&self, place: &mir::Place<'tcx>) -> Ty<'tcx> {
        let tcx = self.cx.tcx;
        let place_ty = place.ty(self.mir, tcx);
        self.monomorphize(&place_ty.to_ty(tcx))
    }
}

