use rustc::mir;
use rustc::ty::layout::{Size, Align};
use rustc::ty::{self, Ty};
use rustc_data_structures::indexed_vec::Idx;

use error::{EvalError, EvalResult};
use eval_context::{EvalContext};
use memory::MemoryPointer;
use value::{PrimVal, Value};

#[derive(Copy, Clone, Debug)]
pub enum Lvalue<'tcx> {
    /// An lvalue referring to a value allocated in the `Memory` system.
    Ptr {
        /// An lvalue may have an invalid (integral or undef) pointer,
        /// since it might be turned back into a reference
        /// before ever being dereferenced.
        ptr: PrimVal,
        extra: LvalueExtra,
    },

    /// An lvalue referring to a value on the stack. Represented by a stack frame index paired with
    /// a Mir local index.
    Local {
        frame: usize,
        local: mir::Local,
    },

    /// An lvalue referring to a global
    Global(GlobalId<'tcx>),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LvalueExtra {
    None,
    Length(u64),
    Vtable(MemoryPointer),
    DowncastVariant(usize),
}

/// Uniquely identifies a specific constant or static.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct GlobalId<'tcx> {
    /// For a constant or static, the `Instance` of the item itself.
    /// For a promoted global, the `Instance` of the function they belong to.
    pub(super) instance: ty::Instance<'tcx>,

    /// The index for promoted globals within their function's `Mir`.
    pub(super) promoted: Option<mir::Promoted>,
}

#[derive(Copy, Clone, Debug)]
pub struct Global<'tcx> {
    pub(super) value: Value,
    /// Only used in `force_allocation` to ensure we don't mark the memory
    /// before the static is initialized. It is possible to convert a
    /// global which initially is `Value::ByVal(PrimVal::Undef)` and gets
    /// lifted to an allocation before the static is fully initialized
    pub(super) initialized: bool,
    pub(super) mutable: bool,
    pub(super) ty: Ty<'tcx>,
}

impl<'tcx> Lvalue<'tcx> {
    /// Produces an Lvalue that will error if attempted to be read from
    pub fn undef() -> Self {
        Self::from_primval_ptr(PrimVal::Undef)
    }

    pub(crate) fn from_primval_ptr(ptr: PrimVal) -> Self {
        Lvalue::Ptr { ptr, extra: LvalueExtra::None }
    }

    pub(crate) fn from_ptr(ptr: MemoryPointer) -> Self {
        Self::from_primval_ptr(PrimVal::Ptr(ptr))
    }

    pub(super) fn to_ptr_and_extra(self) -> (PrimVal, LvalueExtra) {
        match self {
            Lvalue::Ptr { ptr, extra } => (ptr, extra),
            _ => bug!("to_ptr_and_extra: expected Lvalue::Ptr, got {:?}", self),

        }
    }

    pub(super) fn to_ptr(self) -> EvalResult<'tcx, MemoryPointer> {
        let (ptr, extra) = self.to_ptr_and_extra();
        assert_eq!(extra, LvalueExtra::None);
        ptr.to_ptr()
    }

    pub(super) fn elem_ty_and_len(self, ty: Ty<'tcx>) -> (Ty<'tcx>, u64) {
        match ty.sty {
            ty::TyArray(elem, n) => (elem, n as u64),

            ty::TySlice(elem) => {
                match self {
                    Lvalue::Ptr { extra: LvalueExtra::Length(len), .. } => (elem, len),
                    _ => bug!("elem_ty_and_len of a TySlice given non-slice lvalue: {:?}", self),
                }
            }

            _ => bug!("elem_ty_and_len expected array or slice, got {:?}", ty),
        }
    }
}

impl<'tcx> Global<'tcx> {
    pub(super) fn uninitialized(ty: Ty<'tcx>) -> Self {
        Global {
            value: Value::ByVal(PrimVal::Undef),
            mutable: true,
            ty,
            initialized: false,
        }
    }

    pub(super) fn initialized(ty: Ty<'tcx>, value: Value, mutable: bool) -> Self {
        Global {
            value,
            mutable,
            ty,
            initialized: true,
        }
    }
}

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    /// Reads a value from the lvalue without going through the intermediate step of obtaining
    /// a `miri::Lvalue`
    pub fn try_read_lvalue(&mut self, lvalue: &mir::Lvalue<'tcx>) -> EvalResult<'tcx, Option<Value>> {
        use rustc::mir::Lvalue::*;
        match *lvalue {
            // Might allow this in the future, right now there's no way to do this from Rust code anyway
            Local(mir::RETURN_POINTER) => Err(EvalError::ReadFromReturnPointer),
            // Directly reading a local will always succeed
            Local(local) => self.frame().get_local(local).map(Some),
            // Directly reading a static will always succeed
            Static(ref static_) => {
                let instance = ty::Instance::mono(self.tcx, static_.def_id);
                let cid = GlobalId { instance, promoted: None };
                Ok(Some(self.globals.get(&cid).expect("global not cached").value))
            },
            Projection(ref proj) => self.try_read_lvalue_projection(proj),
        }
    }

    fn try_read_lvalue_projection(&mut self, proj: &mir::LvalueProjection<'tcx>) -> EvalResult<'tcx, Option<Value>> {
        use rustc::mir::ProjectionElem::*;
        let base = match self.try_read_lvalue(&proj.base)? {
            Some(base) => base,
            None => return Ok(None),
        };
        let base_ty = self.lvalue_ty(&proj.base);
        match proj.elem {
            Field(field, _) => match (field.index(), base) {
                // the only field of a struct
                (0, Value::ByVal(val)) => Ok(Some(Value::ByVal(val))),
                // split fat pointers, 2 element tuples, ...
                (0...1, Value::ByValPair(a, b)) if self.get_field_count(base_ty)? == 2 => {
                    let val = [a, b][field.index()];
                    Ok(Some(Value::ByVal(val)))
                },
                // the only field of a struct is a fat pointer
                (0, Value::ByValPair(..)) => Ok(Some(base)),
                _ => Ok(None),
            },
            // The NullablePointer cases should work fine, need to take care for normal enums
            Downcast(..) |
            Subslice { .. } |
            // reading index 0 or index 1 from a ByVal or ByVal pair could be optimized
            ConstantIndex { .. } | Index(_) |
            // No way to optimize this projection any better than the normal lvalue path
            Deref => Ok(None),
        }
    }

    pub(super) fn eval_and_read_lvalue(&mut self, lvalue: &mir::Lvalue<'tcx>) -> EvalResult<'tcx, Value> {
        let ty = self.lvalue_ty(lvalue);
        // Shortcut for things like accessing a fat pointer's field,
        // which would otherwise (in the `eval_lvalue` path) require moving a `ByValPair` to memory
        // and returning an `Lvalue::Ptr` to it
        if let Some(val) = self.try_read_lvalue(lvalue)? {
            return Ok(val);
        }
        let lvalue = self.eval_lvalue(lvalue)?;

        if ty.is_never() {
            return Err(EvalError::Unreachable);
        }

        match lvalue {
            Lvalue::Ptr { ptr, extra } => {
                assert_eq!(extra, LvalueExtra::None);
                Ok(Value::ByRef(ptr))
            }
            Lvalue::Local { frame, local } => {
                self.stack[frame].get_local(local)
            }
            Lvalue::Global(cid) => {
                Ok(self.globals.get(&cid).expect("global not cached").value)
            }
        }
    }

    pub(super) fn eval_lvalue(&mut self, mir_lvalue: &mir::Lvalue<'tcx>) -> EvalResult<'tcx, Lvalue<'tcx>> {
        use rustc::mir::Lvalue::*;
        let lvalue = match *mir_lvalue {
            Local(mir::RETURN_POINTER) => self.frame().return_lvalue,
            Local(local) => Lvalue::Local { frame: self.stack.len() - 1, local },

            Static(ref static_) => {
                let instance = ty::Instance::mono(self.tcx, static_.def_id);
                Lvalue::Global(GlobalId { instance, promoted: None })
            }

            Projection(ref proj) => return self.eval_lvalue_projection(proj),
        };

        if log_enabled!(::log::LogLevel::Trace) {
            self.dump_local(lvalue);
        }

        Ok(lvalue)
    }

    pub fn lvalue_field(
        &mut self,
        base: Lvalue<'tcx>,
        field_index: usize,
        base_ty: Ty<'tcx>,
        field_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Lvalue<'tcx>> {
        let base_layout = self.type_layout(base_ty)?;
        use rustc::ty::layout::Layout::*;
        let (offset, packed) = match *base_layout {
            Univariant { ref variant, .. } => {
                (variant.offsets[field_index], variant.packed)
            },

            General { ref variants, .. } => {
                let (_, base_extra) = base.to_ptr_and_extra();
                if let LvalueExtra::DowncastVariant(variant_idx) = base_extra {
                    // +1 for the discriminant, which is field 0
                    (variants[variant_idx].offsets[field_index + 1], variants[variant_idx].packed)
                } else {
                    bug!("field access on enum had no variant index");
                }
            }

            RawNullablePointer { .. } => {
                assert_eq!(field_index, 0);
                return Ok(base);
            }

            StructWrappedNullablePointer { ref nonnull, .. } => {
                (nonnull.offsets[field_index], nonnull.packed)
            }

            UntaggedUnion { .. } => return Ok(base),

            Vector { element, count } => {
                let field = field_index as u64;
                assert!(field < count);
                let elem_size = element.size(&self.tcx.data_layout).bytes();
                (Size::from_bytes(field * elem_size), false)
            }

            // We treat arrays + fixed sized indexing like field accesses
            Array { .. } => {
                let field = field_index as u64;
                let elem_size = match base_ty.sty {
                    ty::TyArray(elem_ty, n) => {
                        assert!(field < n as u64);
                        self.type_size(elem_ty)?.expect("array elements are sized") as u64
                    },
                    _ => bug!("lvalue_field: got Array layout but non-array type {:?}", base_ty),
                };
                (Size::from_bytes(field * elem_size), false)
            }

            FatPointer { .. } => {
                let bytes = field_index as u64 * self.memory.pointer_size();
                let offset = Size::from_bytes(bytes);
                (offset, false)
            }

            _ => bug!("field access on non-product type: {:?}", base_layout),
        };

        // Do not allocate in trivial cases
        let (base_ptr, base_extra) = match base {
            Lvalue::Ptr { ptr, extra } => (ptr, extra),
            Lvalue::Local { frame, local } => match self.stack[frame].get_local(local)? {
                // in case the type has a single field, just return the value
                Value::ByVal(_) if self.get_field_count(base_ty).map(|c| c == 1).unwrap_or(false) => {
                    assert_eq!(offset.bytes(), 0, "ByVal can only have 1 non zst field with offset 0");
                    return Ok(base);
                },
                Value::ByRef(_) |
                Value::ByValPair(..) |
                Value::ByVal(_) => self.force_allocation(base)?.to_ptr_and_extra(),
            },
            Lvalue::Global(cid) => match self.globals.get(&cid).expect("uncached global").value {
                // in case the type has a single field, just return the value
                Value::ByVal(_) if self.get_field_count(base_ty).map(|c| c == 1).unwrap_or(false) => {
                    assert_eq!(offset.bytes(), 0, "ByVal can only have 1 non zst field with offset 0");
                    return Ok(base);
                },
                Value::ByRef(_) |
                Value::ByValPair(..) |
                Value::ByVal(_) => self.force_allocation(base)?.to_ptr_and_extra(),
            },
        };

        let offset = match base_extra {
            LvalueExtra::Vtable(tab) => {
                let (_, align) = self.size_and_align_of_dst(base_ty, Value::ByValPair(base_ptr, PrimVal::Ptr(tab)))?;
                offset.abi_align(Align::from_bytes(align, align).unwrap()).bytes()
            }
            _ => offset.bytes(),
        };

        let ptr = base_ptr.offset(offset, self.memory.layout)?;

        let field_ty = self.monomorphize(field_ty, self.substs());

        if packed {
            let size = self.type_size(field_ty)?.expect("packed struct must be sized");
            self.memory.mark_packed(ptr.to_ptr()?, size);
        }

        let extra = if self.type_is_sized(field_ty) {
            LvalueExtra::None
        } else {
            match base_extra {
                LvalueExtra::None => bug!("expected fat pointer"),
                LvalueExtra::DowncastVariant(..) =>
                    bug!("Rust doesn't support unsized fields in enum variants"),
                LvalueExtra::Vtable(_) |
                LvalueExtra::Length(_) => {},
            }
            base_extra
        };

        Ok(Lvalue::Ptr { ptr, extra })
    }

    fn eval_lvalue_projection(
        &mut self,
        proj: &mir::LvalueProjection<'tcx>,
    ) -> EvalResult<'tcx, Lvalue<'tcx>> {
        use rustc::mir::ProjectionElem::*;
        let (ptr, extra) = match proj.elem {
            Field(field, field_ty) => {
                let base = self.eval_lvalue(&proj.base)?;
                let base_ty = self.lvalue_ty(&proj.base);
                return self.lvalue_field(base, field.index(), base_ty, field_ty);
            }

            Downcast(_, variant) => {
                let base = self.eval_lvalue(&proj.base)?;
                let base_ty = self.lvalue_ty(&proj.base);
                let base_layout = self.type_layout(base_ty)?;
                // FIXME(solson)
                let base = self.force_allocation(base)?;
                let (base_ptr, base_extra) = base.to_ptr_and_extra();

                use rustc::ty::layout::Layout::*;
                let extra = match *base_layout {
                    General { .. } => LvalueExtra::DowncastVariant(variant),
                    RawNullablePointer { .. } | StructWrappedNullablePointer { .. } => base_extra,
                    _ => bug!("variant downcast on non-aggregate: {:?}", base_layout),
                };
                (base_ptr, extra)
            }

            Deref => {
                let base_ty = self.lvalue_ty(&proj.base);
                let val = self.eval_and_read_lvalue(&proj.base)?;

                let pointee_type = match base_ty.sty {
                    ty::TyRawPtr(ref tam) |
                    ty::TyRef(_, ref tam) => tam.ty,
                    ty::TyAdt(def, _) if def.is_box() => base_ty.boxed_ty(),
                    _ => bug!("can only deref pointer types"),
                };

                trace!("deref to {} on {:?}", pointee_type, val);

                match self.tcx.struct_tail(pointee_type).sty {
                    ty::TyDynamic(..) => {
                        let (ptr, vtable) = val.expect_ptr_vtable_pair(&self.memory)?;
                        (ptr, LvalueExtra::Vtable(vtable))
                    },
                    ty::TyStr | ty::TySlice(_) => {
                        let (ptr, len) = val.expect_slice(&self.memory)?;
                        (ptr, LvalueExtra::Length(len))
                    },
                    _ => (val.read_ptr(&self.memory)?, LvalueExtra::None),
                }
            }

            Index(ref operand) => {
                let base = self.eval_lvalue(&proj.base)?;
                let base_ty = self.lvalue_ty(&proj.base);
                // FIXME(solson)
                let base = self.force_allocation(base)?;
                let (base_ptr, _) = base.to_ptr_and_extra();

                let (elem_ty, len) = base.elem_ty_and_len(base_ty);
                let elem_size = self.type_size(elem_ty)?.expect("slice element must be sized");
                let n_ptr = self.eval_operand(operand)?;
                let usize = self.tcx.types.usize;
                let n = self.value_to_primval(n_ptr, usize)?.to_u64()?;
                assert!(n < len, "Tried to access element {} of array/slice with length {}", n, len);
                let ptr = base_ptr.offset(n * elem_size, self.memory.layout)?;
                (ptr, LvalueExtra::None)
            }

            ConstantIndex { offset, min_length, from_end } => {
                let base = self.eval_lvalue(&proj.base)?;
                let base_ty = self.lvalue_ty(&proj.base);
                // FIXME(solson)
                let base = self.force_allocation(base)?;
                let (base_ptr, _) = base.to_ptr_and_extra();

                let (elem_ty, n) = base.elem_ty_and_len(base_ty);
                let elem_size = self.type_size(elem_ty)?.expect("sequence element must be sized");
                assert!(n >= min_length as u64);

                let index = if from_end {
                    n - u64::from(offset)
                } else {
                    u64::from(offset)
                };

                let ptr = base_ptr.offset(index * elem_size, self.memory.layout)?;
                (ptr, LvalueExtra::None)
            }

            Subslice { from, to } => {
                let base = self.eval_lvalue(&proj.base)?;
                let base_ty = self.lvalue_ty(&proj.base);
                // FIXME(solson)
                let base = self.force_allocation(base)?;
                let (base_ptr, _) = base.to_ptr_and_extra();

                let (elem_ty, n) = base.elem_ty_and_len(base_ty);
                let elem_size = self.type_size(elem_ty)?.expect("slice element must be sized");
                assert!(u64::from(from) <= n - u64::from(to));
                let ptr = base_ptr.offset(u64::from(from) * elem_size, self.memory.layout)?;
                let extra = LvalueExtra::Length(n - u64::from(to) - u64::from(from));
                (ptr, extra)
            }
        };

        Ok(Lvalue::Ptr { ptr, extra })
    }

    pub(super) fn lvalue_ty(&self, lvalue: &mir::Lvalue<'tcx>) -> Ty<'tcx> {
        self.monomorphize(lvalue.ty(self.mir(), self.tcx).to_ty(self.tcx), self.substs())
    }
}
