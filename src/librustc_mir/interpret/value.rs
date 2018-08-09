//! Reading and writing values from/to memory, handling LocalValue and the ByRef optimization,
//! reading/writing discriminants

use std::mem;

use rustc::mir;
use rustc::ty::layout::{self, Size, Align, IntegerExt, LayoutOf, TyLayout, Primitive};
use rustc::ty::{self, Ty, TyCtxt, TypeAndMut};
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc::mir::interpret::{
    GlobalId, Value, Scalar, FrameInfo, AllocType,
    EvalResult, EvalErrorKind, Pointer, ConstValue,
    ScalarMaybeUndef,
};

use super::{Place, PlaceExtra, Memory, Frame,
            HasMemory, MemoryKind,
            Machine, ValTy, EvalContext};

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum LocalValue {
    Dead,
    Live(Value),
}

impl LocalValue {
    pub fn access(self) -> EvalResult<'static, Value> {
        match self {
            LocalValue::Dead => err!(DeadLocal),
            LocalValue::Live(val) => Ok(val),
        }
    }
}

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    pub fn write_ptr(&mut self, dest: Place, val: Scalar, dest_ty: Ty<'tcx>) -> EvalResult<'tcx> {
        let valty = ValTy {
            value: val.to_value(),
            ty: dest_ty,
        };
        self.write_value(valty, dest)
    }

    pub fn write_scalar(
        &mut self,
        dest: Place,
        val: impl Into<ScalarMaybeUndef>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        let valty = ValTy {
            value: Value::Scalar(val.into()),
            ty: dest_ty,
        };
        self.write_value(valty, dest)
    }

    pub fn write_value(
        &mut self,
        ValTy { value: src_val, ty: dest_ty } : ValTy<'tcx>,
        dest: Place,
    ) -> EvalResult<'tcx> {
        //trace!("Writing {:?} to {:?} at type {:?}", src_val, dest, dest_ty);
        // Note that it is really important that the type here is the right one, and matches the type things are read at.
        // In case `src_val` is a `ScalarPair`, we don't do any magic here to handle padding properly, which is only
        // correct if we never look at this data with the wrong type.

        match dest {
            Place::Ptr { ptr, align, extra } => {
                assert_eq!(extra, PlaceExtra::None);
                self.write_value_to_ptr(src_val, ptr.unwrap_or_err()?, align, dest_ty)
            }

            Place::Local { frame, local } => {
                let old_val = self.stack[frame].locals[local].access()?;
                self.write_value_possibly_by_val(
                    src_val,
                    |this, val| this.stack[frame].set_local(local, val),
                    old_val,
                    dest_ty,
                )
            }
        }
    }

    // The cases here can be a bit subtle. Read carefully!
    fn write_value_possibly_by_val<F: FnOnce(&mut Self, Value) -> EvalResult<'tcx>>(
        &mut self,
        src_val: Value,
        write_dest: F,
        old_dest_val: Value,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        // FIXME: this should be a layout check, not underlying value
        if let Value::ByRef(dest_ptr, align) = old_dest_val {
            // If the value is already `ByRef` (that is, backed by an `Allocation`),
            // then we must write the new value into this allocation, because there may be
            // other pointers into the allocation. These other pointers are logically
            // pointers into the local variable, and must be able to observe the change.
            //
            // Thus, it would be an error to replace the `ByRef` with a `ByVal`, unless we
            // knew for certain that there were no outstanding pointers to this allocation.
            self.write_value_to_ptr(src_val, dest_ptr, align, dest_ty)?;
        } else if let Value::ByRef(src_ptr, align) = src_val {
            // If the value is not `ByRef`, then we know there are no pointers to it
            // and we can simply overwrite the `Value` in the locals array directly.
            //
            // In this specific case, where the source value is `ByRef`, we must duplicate
            // the allocation, because this is a by-value operation. It would be incorrect
            // if they referred to the same allocation, since then a change to one would
            // implicitly change the other.
            //
            // It is a valid optimization to attempt reading a primitive value out of the
            // source and write that into the destination without making an allocation, so
            // we do so here.
            if let Ok(Some(src_val)) = self.try_read_value(src_ptr, align, dest_ty) {
                write_dest(self, src_val)?;
            } else {
                let layout = self.layout_of(dest_ty)?;
                let dest_ptr = self.alloc_ptr(layout)?.into();
                self.memory.copy(src_ptr, align.min(layout.align), dest_ptr, layout.align, layout.size, false)?;
                write_dest(self, Value::ByRef(dest_ptr, layout.align))?;
            }
        } else {
            // Finally, we have the simple case where neither source nor destination are
            // `ByRef`. We may simply copy the source value over the the destintion.
            write_dest(self, src_val)?;
        }
        Ok(())
    }

    pub fn write_value_to_ptr(
        &mut self,
        value: Value,
        dest: Scalar,
        dest_align: Align,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        let layout = self.layout_of(dest_ty)?;
        trace!("write_value_to_ptr: {:#?}, {}, {:#?}", value, dest_ty, layout);
        match value {
            Value::ByRef(ptr, align) => {
                self.memory.copy(ptr, align.min(layout.align), dest, dest_align.min(layout.align), layout.size, false)
            }
            Value::Scalar(scalar) => {
                let signed = match layout.abi {
                    layout::Abi::Scalar(ref scal) => match scal.value {
                        layout::Primitive::Int(_, signed) => signed,
                        _ => false,
                    },
                    _ => false,
                };
                self.memory.write_scalar(dest, dest_align, scalar, layout.size, layout.align, signed)
            }
            Value::ScalarPair(a_val, b_val) => {
                trace!("write_value_to_ptr valpair: {:#?}", layout);
                let (a, b) = match layout.abi {
                    layout::Abi::ScalarPair(ref a, ref b) => (&a.value, &b.value),
                    _ => bug!("write_value_to_ptr: invalid ScalarPair layout: {:#?}", layout)
                };
                let (a_size, b_size) = (a.size(&self), b.size(&self));
                let (a_align, b_align) = (a.align(&self), b.align(&self));
                let a_ptr = dest;
                let b_offset = a_size.abi_align(b_align);
                let b_ptr = dest.ptr_offset(b_offset, &self)?.into();
                // TODO: What about signedess?
                self.memory.write_scalar(a_ptr, dest_align, a_val, a_size, a_align, false)?;
                self.memory.write_scalar(b_ptr, dest_align, b_val, b_size, b_align, false)
            }
        }
    }

    pub fn try_read_value(&self, ptr: Scalar, ptr_align: Align, ty: Ty<'tcx>) -> EvalResult<'tcx, Option<Value>> {
        let layout = self.layout_of(ty)?;
        self.memory.check_align(ptr, ptr_align)?;

        if layout.size.bytes() == 0 {
            return Ok(Some(Value::Scalar(ScalarMaybeUndef::Scalar(Scalar::Bits { bits: 0, size: 0 }))));
        }

        let ptr = ptr.to_ptr()?;

        match layout.abi {
            layout::Abi::Scalar(..) => {
                let scalar = self.memory.read_scalar(ptr, ptr_align, layout.size)?;
                Ok(Some(Value::Scalar(scalar)))
            }
            layout::Abi::ScalarPair(ref a, ref b) => {
                let (a, b) = (&a.value, &b.value);
                let (a_size, b_size) = (a.size(self), b.size(self));
                let a_ptr = ptr;
                let b_offset = a_size.abi_align(b.align(self));
                let b_ptr = ptr.offset(b_offset, self)?.into();
                let a_val = self.memory.read_scalar(a_ptr, ptr_align, a_size)?;
                let b_val = self.memory.read_scalar(b_ptr, ptr_align, b_size)?;
                Ok(Some(Value::ScalarPair(a_val, b_val)))
            }
            _ => Ok(None),
        }
    }

    pub fn read_value(&self, ptr: Scalar, align: Align, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        if let Some(val) = self.try_read_value(ptr, align, ty)? {
            Ok(val)
        } else {
            bug!("primitive read failed for type: {:?}", ty);
        }
    }

    pub(super) fn eval_operand_to_scalar(
        &mut self,
        op: &mir::Operand<'tcx>,
    ) -> EvalResult<'tcx, Scalar> {
        let valty = self.eval_operand(op)?;
        self.value_to_scalar(valty)
    }

    pub(crate) fn operands_to_args(
        &mut self,
        ops: &[mir::Operand<'tcx>],
    ) -> EvalResult<'tcx, Vec<ValTy<'tcx>>> {
        ops.into_iter()
            .map(|op| self.eval_operand(op))
            .collect()
    }

    pub fn eval_operand(&mut self, op: &mir::Operand<'tcx>) -> EvalResult<'tcx, ValTy<'tcx>> {
        use rustc::mir::Operand::*;
        let ty = self.monomorphize(op.ty(self.mir(), *self.tcx), self.substs());
        match *op {
            // FIXME: do some more logic on `move` to invalidate the old location
            Copy(ref place) |
            Move(ref place) => {
                Ok(ValTy {
                    value: self.eval_and_read_place(place)?,
                    ty
                })
            },

            Constant(ref constant) => {
                let value = self.const_to_value(constant.literal.val)?;

                Ok(ValTy {
                    value,
                    ty,
                })
            }
        }
    }

    pub fn deallocate_local(&mut self, local: LocalValue) -> EvalResult<'tcx> {
        // FIXME: should we tell the user that there was a local which was never written to?
        if let LocalValue::Live(Value::ByRef(ptr, _align)) = local {
            trace!("deallocating local");
            let ptr = ptr.to_ptr()?;
            self.memory.dump_alloc(ptr.alloc_id);
            self.memory.deallocate_local(ptr)?;
        };
        Ok(())
    }

    pub fn allocate_place_for_value(
        &mut self,
        value: Value,
        layout: TyLayout<'tcx>,
        variant: Option<usize>,
    ) -> EvalResult<'tcx, Place> {
        let (ptr, align) = match value {
            Value::ByRef(ptr, align) => (ptr, align),
            Value::ScalarPair(..) | Value::Scalar(_) => {
                let ptr = self.alloc_ptr(layout)?.into();
                self.write_value_to_ptr(value, ptr, layout.align, layout.ty)?;
                (ptr, layout.align)
            },
        };
        Ok(Place::Ptr {
            ptr: ptr.into(),
            align,
            extra: variant.map_or(PlaceExtra::None, PlaceExtra::DowncastVariant),
        })
    }

    pub fn force_allocation(&mut self, place: Place) -> EvalResult<'tcx, Place> {
        let new_place = match place {
            Place::Local { frame, local } => {
                match self.stack[frame].locals[local].access()? {
                    Value::ByRef(ptr, align) => {
                        Place::Ptr {
                            ptr: ptr.into(),
                            align,
                            extra: PlaceExtra::None,
                        }
                    }
                    val => {
                        let ty = self.stack[frame].mir.local_decls[local].ty;
                        let ty = self.monomorphize(ty, self.stack[frame].instance.substs);
                        let layout = self.layout_of(ty)?;
                        let ptr = self.alloc_ptr(layout)?;
                        self.stack[frame].locals[local] =
                            LocalValue::Live(Value::ByRef(ptr.into(), layout.align)); // it stays live

                        let place = Place::from_ptr(ptr, layout.align);
                        self.write_value(ValTy { value: val, ty }, place)?;
                        place
                    }
                }
            }
            Place::Ptr { .. } => place,
        };
        Ok(new_place)
    }

    /// Convert to ByVal or ScalarPair *if possible*, leave `ByRef` otherwise
    pub fn try_read_by_ref(&self, mut val: Value, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        if let Value::ByRef(ptr, align) = val {
            if let Some(read_val) = self.try_read_value(ptr, align, ty)? {
                val = read_val;
            }
        }
        Ok(val)
    }

    pub fn value_to_scalar(
        &self,
        ValTy { value, ty } : ValTy<'tcx>,
    ) -> EvalResult<'tcx, Scalar> {
        let value = match value {
            Value::ByRef(ptr, align) => self.read_value(ptr, align, ty)?,
            scalar_or_pair => scalar_or_pair,
        };
        match value {
            Value::ByRef(..) => bug!("read_value can't result in `ByRef`"),

            Value::Scalar(scalar) => scalar.unwrap_or_err(),

            Value::ScalarPair(..) => bug!("value_to_scalar can't work with fat pointers"),
        }
    }

    pub fn storage_live(&mut self, local: mir::Local) -> EvalResult<'tcx, LocalValue> {
        trace!("{:?} is now live", local);

        let ty = self.frame().mir.local_decls[local].ty;
        let init = self.init_value(ty)?;
        // StorageLive *always* kills the value that's currently stored
        Ok(mem::replace(&mut self.frame_mut().locals[local], LocalValue::Live(init)))
    }

    pub(super) fn init_value(&mut self, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        let ty = self.monomorphize(ty, self.substs());
        let layout = self.layout_of(ty)?;
        Ok(match layout.abi {
            layout::Abi::Scalar(..) => Value::Scalar(ScalarMaybeUndef::Undef),
            layout::Abi::ScalarPair(..) => Value::ScalarPair(
                ScalarMaybeUndef::Undef,
                ScalarMaybeUndef::Undef,
            ),
            _ => Value::ByRef(self.alloc_ptr(layout)?.into(), layout.align),
        })
    }

    /// reads a tag and produces the corresponding variant index
    pub fn read_discriminant_as_variant_index(
        &self,
        place: Place,
        layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx, usize> {
        match layout.variants {
            ty::layout::Variants::Single { index } => Ok(index),
            ty::layout::Variants::Tagged { .. } => {
                let discr_val = self.read_discriminant_value(place, layout)?;
                layout
                    .ty
                    .ty_adt_def()
                    .expect("tagged layout for non adt")
                    .discriminants(self.tcx.tcx)
                    .position(|var| var.val == discr_val)
                    .ok_or_else(|| EvalErrorKind::InvalidDiscriminant.into())
            }
            ty::layout::Variants::NicheFilling { .. } => {
                let discr_val = self.read_discriminant_value(place, layout)?;
                assert_eq!(discr_val as usize as u128, discr_val);
                Ok(discr_val as usize)
            },
        }
    }

    pub fn read_discriminant_value(
        &self,
        place: Place,
        layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx, u128> {
        trace!("read_discriminant_value {:#?}", layout);
        if layout.abi == layout::Abi::Uninhabited {
            return Ok(0);
        }

        match layout.variants {
            layout::Variants::Single { index } => {
                let discr_val = layout.ty.ty_adt_def().map_or(
                    index as u128,
                    |def| def.discriminant_for_variant(*self.tcx, index).val);
                return Ok(discr_val);
            }
            layout::Variants::Tagged { .. } |
            layout::Variants::NicheFilling { .. } => {},
        }
        let discr_place_val = self.read_place(place)?;
        let (discr_val, discr) = self.read_field(discr_place_val, None, mir::Field::new(0), layout)?;
        trace!("discr value: {:?}, {:?}", discr_val, discr);
        let raw_discr = self.value_to_scalar(ValTy {
            value: discr_val,
            ty: discr.ty
        })?;
        let discr_val = match layout.variants {
            layout::Variants::Single { .. } => bug!(),
            // FIXME: should we catch invalid discriminants here?
            layout::Variants::Tagged { .. } => {
                if discr.ty.is_signed() {
                    let i = raw_discr.to_bits(discr.size)? as i128;
                    // going from layout tag type to typeck discriminant type
                    // requires first sign extending with the layout discriminant
                    let shift = 128 - discr.size.bits();
                    let sexted = (i << shift) >> shift;
                    // and then zeroing with the typeck discriminant type
                    let discr_ty = layout
                        .ty
                        .ty_adt_def().expect("tagged layout corresponds to adt")
                        .repr
                        .discr_type();
                    let discr_ty = layout::Integer::from_attr(self.tcx.tcx, discr_ty);
                    let shift = 128 - discr_ty.size().bits();
                    let truncatee = sexted as u128;
                    (truncatee << shift) >> shift
                } else {
                    raw_discr.to_bits(discr.size)?
                }
            },
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                let variants_start = *niche_variants.start() as u128;
                let variants_end = *niche_variants.end() as u128;
                match raw_discr {
                    Scalar::Ptr(_) => {
                        assert!(niche_start == 0);
                        assert!(variants_start == variants_end);
                        dataful_variant as u128
                    },
                    Scalar::Bits { bits: raw_discr, size } => {
                        assert_eq!(size as u64, discr.size.bytes());
                        let discr = raw_discr.wrapping_sub(niche_start)
                            .wrapping_add(variants_start);
                        if variants_start <= discr && discr <= variants_end {
                            discr
                        } else {
                            dataful_variant as u128
                        }
                    },
                }
            }
        };

        Ok(discr_val)
    }


    pub fn write_discriminant_value(
        &mut self,
        dest_ty: Ty<'tcx>,
        dest: Place,
        variant_index: usize,
    ) -> EvalResult<'tcx> {
        let layout = self.layout_of(dest_ty)?;

        match layout.variants {
            layout::Variants::Single { index } => {
                if index != variant_index {
                    // If the layout of an enum is `Single`, all
                    // other variants are necessarily uninhabited.
                    assert_eq!(layout.for_variant(&self, variant_index).abi,
                               layout::Abi::Uninhabited);
                }
            }
            layout::Variants::Tagged { ref tag, .. } => {
                let discr_val = dest_ty.ty_adt_def().unwrap()
                    .discriminant_for_variant(*self.tcx, variant_index)
                    .val;

                // raw discriminants for enums are isize or bigger during
                // their computation, but the in-memory tag is the smallest possible
                // representation
                let size = tag.value.size(self.tcx.tcx);
                let shift = 128 - size.bits();
                let discr_val = (discr_val << shift) >> shift;

                let (discr_dest, tag) = self.place_field(dest, mir::Field::new(0), layout)?;
                self.write_scalar(discr_dest, Scalar::Bits {
                    bits: discr_val,
                    size: size.bytes() as u8,
                }, tag.ty)?;
            }
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                if variant_index != dataful_variant {
                    let (niche_dest, niche) =
                        self.place_field(dest, mir::Field::new(0), layout)?;
                    let niche_value = ((variant_index - niche_variants.start()) as u128)
                        .wrapping_add(niche_start);
                    self.write_scalar(niche_dest, Scalar::Bits {
                        bits: niche_value,
                        size: niche.size.bytes() as u8,
                    }, niche.ty)?;
                }
            }
        }

        Ok(())
    }

    pub fn str_to_value(&mut self, s: &str) -> EvalResult<'tcx, Value> {
        let ptr = self.memory.allocate_bytes(s.as_bytes());
        Ok(Scalar::Ptr(ptr).to_value_with_len(s.len() as u64, self.tcx.tcx))
    }

    pub fn const_to_value(
        &mut self,
        val: ConstValue<'tcx>,
    ) -> EvalResult<'tcx, Value> {
        match val {
            ConstValue::Unevaluated(def_id, substs) => {
                let instance = self.resolve(def_id, substs)?;
                self.read_global_as_value(GlobalId {
                    instance,
                    promoted: None,
                })
            }
            ConstValue::ByRef(alloc, offset) => {
                // FIXME: Allocate new AllocId for all constants inside
                let id = self.memory.allocate_value(alloc.clone(), MemoryKind::Stack)?;
                Ok(Value::ByRef(Pointer::new(id, offset).into(), alloc.align))
            },
            ConstValue::ScalarPair(a, b) => Ok(Value::ScalarPair(a.into(), b.into())),
            ConstValue::Scalar(val) => Ok(Value::Scalar(val.into())),
        }
    }
}

impl<'mir, 'tcx> Frame<'mir, 'tcx> {
    pub(super) fn set_local(&mut self, local: mir::Local, value: Value) -> EvalResult<'tcx> {
        match self.locals[local] {
            LocalValue::Dead => err!(DeadLocal),
            LocalValue::Live(ref mut local) => {
                *local = value;
                Ok(())
            }
        }
    }

    /// Returns the old value of the local
    pub fn storage_dead(&mut self, local: mir::Local) -> LocalValue {
        trace!("{:?} is now dead", local);

        mem::replace(&mut self.locals[local], LocalValue::Dead)
    }
}
