use rustc::mir;
use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, Align, LayoutOf, TyLayout};
use rustc_data_structures::indexed_vec::Idx;

use rustc::mir::interpret::{GlobalId, Value, PrimVal, EvalResult, Pointer, MemoryPointer};
use super::{EvalContext, Machine, ValTy};
use interpret::memory::HasMemory;

#[derive(Copy, Clone, Debug)]
pub enum Place {
    /// A place referring to a value allocated in the `Memory` system.
    Ptr {
        /// A place may have an invalid (integral or undef) pointer,
        /// since it might be turned back into a reference
        /// before ever being dereferenced.
        ptr: Pointer,
        align: Align,
        extra: PlaceExtra,
    },

    /// A place referring to a value on the stack. Represented by a stack frame index paired with
    /// a Mir local index.
    Local { frame: usize, local: mir::Local },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PlaceExtra {
    None,
    Length(u64),
    Vtable(MemoryPointer),
    DowncastVariant(usize),
}

impl<'tcx> Place {
    /// Produces a Place that will error if attempted to be read from
    pub fn undef() -> Self {
        Self::from_primval_ptr(PrimVal::Undef.into(), Align::from_bytes(1, 1).unwrap())
    }

    pub fn from_primval_ptr(ptr: Pointer, align: Align) -> Self {
        Place::Ptr {
            ptr,
            align,
            extra: PlaceExtra::None,
        }
    }

    pub fn from_ptr(ptr: MemoryPointer, align: Align) -> Self {
        Self::from_primval_ptr(ptr.into(), align)
    }

    pub fn to_ptr_align_extra(self) -> (Pointer, Align, PlaceExtra) {
        match self {
            Place::Ptr { ptr, align, extra } => (ptr, align, extra),
            _ => bug!("to_ptr_and_extra: expected Place::Ptr, got {:?}", self),

        }
    }

    pub fn to_ptr_align(self) -> (Pointer, Align) {
        let (ptr, align, _extra) = self.to_ptr_align_extra();
        (ptr, align)
    }

    pub fn to_ptr(self) -> EvalResult<'tcx, MemoryPointer> {
        // At this point, we forget about the alignment information -- the place has been turned into a reference,
        // and no matter where it came from, it now must be aligned.
        self.to_ptr_align().0.to_ptr()
    }

    pub(super) fn elem_ty_and_len(self, ty: Ty<'tcx>) -> (Ty<'tcx>, u64) {
        match ty.sty {
            ty::TyArray(elem, n) => (elem, n.val.unwrap_u64() as u64),

            ty::TySlice(elem) => {
                match self {
                    Place::Ptr { extra: PlaceExtra::Length(len), .. } => (elem, len),
                    _ => {
                        bug!(
                            "elem_ty_and_len of a TySlice given non-slice place: {:?}",
                            self
                        )
                    }
                }
            }

            _ => bug!("elem_ty_and_len expected array or slice, got {:?}", ty),
        }
    }
}

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    /// Reads a value from the place without going through the intermediate step of obtaining
    /// a `miri::Place`
    pub fn try_read_place(
        &mut self,
        place: &mir::Place<'tcx>,
    ) -> EvalResult<'tcx, Option<Value>> {
        use rustc::mir::Place::*;
        match *place {
            // Might allow this in the future, right now there's no way to do this from Rust code anyway
            Local(mir::RETURN_PLACE) => err!(ReadFromReturnPointer),
            // Directly reading a local will always succeed
            Local(local) => self.frame().get_local(local).map(Some),
            // No fast path for statics. Reading from statics is rare and would require another
            // Machine function to handle differently in miri.
            Static(_) => Ok(None),
            Projection(ref proj) => self.try_read_place_projection(proj),
        }
    }

    pub fn read_field(
        &self,
        base: Value,
        variant: Option<usize>,
        field: mir::Field,
        base_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(Value, Ty<'tcx>)>> {
        let mut base_layout = self.layout_of(base_ty)?;
        if let Some(variant_index) = variant {
            base_layout = base_layout.for_variant(self, variant_index);
        }
        let field_index = field.index();
        let field = base_layout.field(self, field_index)?;
        let offset = base_layout.fields.offset(field_index);
        match base {
            // the field covers the entire type
            Value::ByValPair(..) |
            Value::ByVal(_) if offset.bytes() == 0 && field.size == base_layout.size => Ok(Some((base, field.ty))),
            // split fat pointers, 2 element tuples, ...
            Value::ByValPair(a, b) if base_layout.fields.count() == 2 => {
                let val = [a, b][field_index];
                Ok(Some((Value::ByVal(val), field.ty)))
            },
            _ => Ok(None),
        }
    }

    fn try_read_place_projection(
        &mut self,
        proj: &mir::PlaceProjection<'tcx>,
    ) -> EvalResult<'tcx, Option<Value>> {
        use rustc::mir::ProjectionElem::*;
        let base = match self.try_read_place(&proj.base)? {
            Some(base) => base,
            None => return Ok(None),
        };
        let base_ty = self.place_ty(&proj.base);
        match proj.elem {
            Field(field, _) => Ok(self.read_field(base, None, field, base_ty)?.map(|(f, _)| f)),
            // The NullablePointer cases should work fine, need to take care for normal enums
            Downcast(..) |
            Subslice { .. } |
            // reading index 0 or index 1 from a ByVal or ByVal pair could be optimized
            ConstantIndex { .. } | Index(_) |
            // No way to optimize this projection any better than the normal place path
            Deref => Ok(None),
        }
    }

    /// Returns a value and (in case of a ByRef) if we are supposed to use aligned accesses.
    pub(super) fn eval_and_read_place(
        &mut self,
        place: &mir::Place<'tcx>,
    ) -> EvalResult<'tcx, Value> {
        // Shortcut for things like accessing a fat pointer's field,
        // which would otherwise (in the `eval_place` path) require moving a `ByValPair` to memory
        // and returning an `Place::Ptr` to it
        if let Some(val) = self.try_read_place(place)? {
            return Ok(val);
        }
        let place = self.eval_place(place)?;
        self.read_place(place)
    }

    pub fn read_place(&self, place: Place) -> EvalResult<'tcx, Value> {
        match place {
            Place::Ptr { ptr, align, extra } => {
                assert_eq!(extra, PlaceExtra::None);
                Ok(Value::ByRef(ptr, align))
            }
            Place::Local { frame, local } => self.stack[frame].get_local(local),
        }
    }

    pub fn eval_place(&mut self, mir_place: &mir::Place<'tcx>) -> EvalResult<'tcx, Place> {
        use rustc::mir::Place::*;
        let place = match *mir_place {
            Local(mir::RETURN_PLACE) => self.frame().return_place,
            Local(local) => Place::Local {
                frame: self.cur_frame(),
                local,
            },

            Static(ref static_) => {
                let alloc = self
                    .tcx
                    .interpret_interner
                    .borrow()
                    .get_cached(static_.def_id);
                let layout = self.layout_of(self.place_ty(mir_place))?;
                if let Some(alloc) = alloc {
                    Place::Ptr {
                        ptr: MemoryPointer::new(alloc, 0).into(),
                        align: layout.align,
                        extra: PlaceExtra::None,
                    }
                } else {
                    let instance = ty::Instance::mono(self.tcx, static_.def_id);
                    let cid = GlobalId {
                        instance,
                        promoted: None
                    };
                    let alloc = Machine::init_static(self, cid)?;
                    Place::Ptr {
                        ptr: MemoryPointer::new(alloc, 0).into(),
                        align: layout.align,
                        extra: PlaceExtra::None,
                    }
                }
            }

            Projection(ref proj) => {
                let ty = self.place_ty(&proj.base);
                let place = self.eval_place(&proj.base)?;
                return self.eval_place_projection(place, ty, &proj.elem);
            }
        };

        if log_enabled!(::log::Level::Trace) {
            self.dump_local(place);
        }

        Ok(place)
    }

    pub fn place_field(
        &mut self,
        base: Place,
        field: mir::Field,
        mut base_layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx, (Place, TyLayout<'tcx>)> {
        match base {
            Place::Ptr { extra: PlaceExtra::DowncastVariant(variant_index), .. } => {
                base_layout = base_layout.for_variant(&self, variant_index);
            }
            _ => {}
        }
        let field_index = field.index();
        let field = base_layout.field(&self, field_index)?;
        let offset = base_layout.fields.offset(field_index);

        // Do not allocate in trivial cases
        let (base_ptr, base_align, base_extra) = match base {
            Place::Ptr { ptr, align, extra } => (ptr, align, extra),
            Place::Local { frame, local } => {
                match (&self.stack[frame].get_local(local)?, &base_layout.abi) {
                    // in case the field covers the entire type, just return the value
                    (&Value::ByVal(_), &layout::Abi::Scalar(_)) |
                    (&Value::ByValPair(..), &layout::Abi::ScalarPair(..))
                        if offset.bytes() == 0 && field.size == base_layout.size =>
                    {
                        return Ok((base, field));
                    }
                    _ => self.force_allocation(base)?.to_ptr_align_extra(),
                }
            }
        };

        let offset = match base_extra {
            PlaceExtra::Vtable(tab) => {
                let (_, align) = self.size_and_align_of_dst(
                    base_layout.ty,
                    base_ptr.to_value_with_vtable(tab),
                )?;
                offset.abi_align(align).bytes()
            }
            _ => offset.bytes(),
        };

        let ptr = base_ptr.offset(offset, &self)?;
        let align = base_align.min(base_layout.align).min(field.align);
        let extra = if !field.is_unsized() {
            PlaceExtra::None
        } else {
            match base_extra {
                PlaceExtra::None => bug!("expected fat pointer"),
                PlaceExtra::DowncastVariant(..) => {
                    bug!("Rust doesn't support unsized fields in enum variants")
                }
                PlaceExtra::Vtable(_) |
                PlaceExtra::Length(_) => {}
            }
            base_extra
        };

        Ok((Place::Ptr { ptr, align, extra }, field))
    }

    pub fn val_to_place(&self, val: Value, ty: Ty<'tcx>) -> EvalResult<'tcx, Place> {
        let layout = self.layout_of(ty)?;
        Ok(match self.tcx.struct_tail(ty).sty {
            ty::TyDynamic(..) => {
                let (ptr, vtable) = self.into_ptr_vtable_pair(val)?;
                Place::Ptr {
                    ptr,
                    align: layout.align,
                    extra: PlaceExtra::Vtable(vtable),
                }
            }
            ty::TyStr | ty::TySlice(_) => {
                let (ptr, len) = self.into_slice(val)?;
                Place::Ptr {
                    ptr,
                    align: layout.align,
                    extra: PlaceExtra::Length(len),
                }
            }
            _ => Place::from_primval_ptr(self.into_ptr(val)?, layout.align),
        })
    }

    pub fn place_index(
        &mut self,
        base: Place,
        outer_ty: Ty<'tcx>,
        n: u64,
    ) -> EvalResult<'tcx, Place> {
        // Taking the outer type here may seem odd; it's needed because for array types, the outer type gives away the length.
        let base = self.force_allocation(base)?;
        let (base_ptr, align) = base.to_ptr_align();

        let (elem_ty, len) = base.elem_ty_and_len(outer_ty);
        let elem_size = self.layout_of(elem_ty)?.size.bytes();
        assert!(
            n < len,
            "Tried to access element {} of array/slice with length {}",
            n,
            len
        );
        let ptr = base_ptr.offset(n * elem_size, &*self)?;
        Ok(Place::Ptr {
            ptr,
            align,
            extra: PlaceExtra::None,
        })
    }

    pub(super) fn place_downcast(
        &mut self,
        base: Place,
        variant: usize,
    ) -> EvalResult<'tcx, Place> {
        // FIXME(solson)
        let base = self.force_allocation(base)?;
        let (ptr, align) = base.to_ptr_align();
        let extra = PlaceExtra::DowncastVariant(variant);
        Ok(Place::Ptr { ptr, align, extra })
    }

    pub fn eval_place_projection(
        &mut self,
        base: Place,
        base_ty: Ty<'tcx>,
        proj_elem: &mir::ProjectionElem<'tcx, mir::Local, Ty<'tcx>>,
    ) -> EvalResult<'tcx, Place> {
        use rustc::mir::ProjectionElem::*;
        match *proj_elem {
            Field(field, _) => {
                let layout = self.layout_of(base_ty)?;
                Ok(self.place_field(base, field, layout)?.0)
            }

            Downcast(_, variant) => {
                self.place_downcast(base, variant)
            }

            Deref => {
                let val = self.read_place(base)?;

                let pointee_type = match base_ty.sty {
                    ty::TyRawPtr(ref tam) |
                    ty::TyRef(_, ref tam) => tam.ty,
                    ty::TyAdt(def, _) if def.is_box() => base_ty.boxed_ty(),
                    _ => bug!("can only deref pointer types"),
                };

                trace!("deref to {} on {:?}", pointee_type, val);

                self.val_to_place(val, pointee_type)
            }

            Index(local) => {
                let value = self.frame().get_local(local)?;
                let ty = self.tcx.types.usize;
                let n = self.value_to_primval(ValTy { value, ty })?.to_u64()?;
                self.place_index(base, base_ty, n)
            }

            ConstantIndex {
                offset,
                min_length,
                from_end,
            } => {
                // FIXME(solson)
                let base = self.force_allocation(base)?;
                let (base_ptr, align) = base.to_ptr_align();

                let (elem_ty, n) = base.elem_ty_and_len(base_ty);
                let elem_size = self.layout_of(elem_ty)?.size.bytes();
                assert!(n >= min_length as u64);

                let index = if from_end {
                    n - u64::from(offset)
                } else {
                    u64::from(offset)
                };

                let ptr = base_ptr.offset(index * elem_size, &self)?;
                Ok(Place::Ptr { ptr, align, extra: PlaceExtra::None })
            }

            Subslice { from, to } => {
                // FIXME(solson)
                let base = self.force_allocation(base)?;
                let (base_ptr, align) = base.to_ptr_align();

                let (elem_ty, n) = base.elem_ty_and_len(base_ty);
                let elem_size = self.layout_of(elem_ty)?.size.bytes();
                assert!(u64::from(from) <= n - u64::from(to));
                let ptr = base_ptr.offset(u64::from(from) * elem_size, &self)?;
                // sublicing arrays produces arrays
                let extra = if self.type_is_sized(base_ty) {
                    PlaceExtra::None
                } else {
                    PlaceExtra::Length(n - u64::from(to) - u64::from(from))
                };
                Ok(Place::Ptr { ptr, align, extra })
            }
        }
    }

    pub fn place_ty(&self, place: &mir::Place<'tcx>) -> Ty<'tcx> {
        self.monomorphize(
            place.ty(self.mir(), self.tcx).to_ty(self.tcx),
            self.substs(),
        )
    }
}
