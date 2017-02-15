use rustc::hir::def_id::DefId;
use rustc::mir;
use rustc::ty::layout::{Size, Align};
use rustc::ty::subst::Substs;
use rustc::ty::{self, Ty};
use rustc_data_structures::indexed_vec::Idx;

use error::EvalResult;
use eval_context::{EvalContext};
use memory::Pointer;
use value::{PrimVal, Value};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Lvalue<'tcx> {
    /// An lvalue referring to a value allocated in the `Memory` system.
    Ptr {
        ptr: Pointer,
        extra: LvalueExtra,
    },

    /// An lvalue referring to a value on the stack. Represented by a stack frame index paired with
    /// a Mir local index.
    Local {
        frame: usize,
        local: mir::Local,
        /// Optionally, this lvalue can point to a field of the stack value
        field: Option<(usize, Ty<'tcx>)>,
    },

    /// An lvalue referring to a global
    Global(GlobalId<'tcx>),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LvalueExtra {
    None,
    Length(u64),
    Vtable(Pointer),
    DowncastVariant(usize),
}

/// Uniquely identifies a specific constant or static.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct GlobalId<'tcx> {
    /// For a constant or static, the `DefId` of the item itself.
    /// For a promoted global, the `DefId` of the function they belong to.
    pub(super) def_id: DefId,

    /// For statics and constants this is `Substs::empty()`, so only promoteds and associated
    /// constants actually have something useful here. We could special case statics and constants,
    /// but that would only require more branching when working with constants, and not bring any
    /// real benefits.
    pub(super) substs: &'tcx Substs<'tcx>,

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
    pub fn from_ptr(ptr: Pointer) -> Self {
        Lvalue::Ptr { ptr, extra: LvalueExtra::None }
    }

    pub(super) fn to_ptr_and_extra(self) -> (Pointer, LvalueExtra) {
        match self {
            Lvalue::Ptr { ptr, extra } => (ptr, extra),
            _ => bug!("to_ptr_and_extra: expected Lvalue::Ptr, got {:?}", self),

        }
    }

    pub(super) fn to_ptr(self) -> Pointer {
        let (ptr, extra) = self.to_ptr_and_extra();
        assert_eq!(extra, LvalueExtra::None);
        ptr
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
}

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub(super) fn eval_and_read_lvalue(&mut self, lvalue: &mir::Lvalue<'tcx>) -> EvalResult<'tcx, Value> {
        let lvalue = self.eval_lvalue(lvalue)?;
        Ok(self.read_lvalue(lvalue))
    }

    pub fn read_lvalue(&self, lvalue: Lvalue<'tcx>) -> Value {
        match lvalue {
            Lvalue::Ptr { ptr, extra } => {
                assert_eq!(extra, LvalueExtra::None);
                Value::ByRef(ptr)
            }
            Lvalue::Local { frame, local, field } => self.stack[frame].get_local(local, field.map(|(i, _)| i)),
            Lvalue::Global(cid) => self.globals.get(&cid).expect("global not cached").value,
        }
    }

    pub(super) fn eval_lvalue(&mut self, mir_lvalue: &mir::Lvalue<'tcx>) -> EvalResult<'tcx, Lvalue<'tcx>> {
        use rustc::mir::Lvalue::*;
        let lvalue = match *mir_lvalue {
            Local(mir::RETURN_POINTER) => self.frame().return_lvalue,
            Local(local) => Lvalue::Local { frame: self.stack.len() - 1, local, field: None },

            Static(def_id) => {
                let substs = self.tcx.intern_substs(&[]);
                Lvalue::Global(GlobalId { def_id, substs, promoted: None })
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

            _ => bug!("field access on non-product type: {:?}", base_layout),
        };

        let (base_ptr, base_extra) = match base {
            Lvalue::Ptr { ptr, extra } => (ptr, extra),
            Lvalue::Local { frame, local, field } => match self.stack[frame].get_local(local, field.map(|(i, _)| i)) {
                Value::ByRef(ptr) => {
                    assert!(field.is_none(), "local can't be ByRef and have a field offset");
                    (ptr, LvalueExtra::None)
                },
                Value::ByVal(PrimVal::Undef) => {
                    // FIXME: allocate in fewer cases
                    if self.ty_to_primval_kind(base_ty).is_ok() {
                        return Ok(base);
                    } else {
                        (self.force_allocation(base)?.to_ptr(), LvalueExtra::None)
                    }
                },
                Value::ByVal(_) => {
                    assert_eq!(field_index, 0, "ByVal can only have 1 non zst field with offset 0");
                    return Ok(base);
                },
                Value::ByValPair(_, _) => {
                    let field_count = self.get_field_count(base_ty)?;
                    if field_count == 1 {
                        assert_eq!(field_index, 0, "{:?} has only one field", base_ty);
                        return Ok(base);
                    }
                    assert_eq!(field_count, 2);
                    assert!(field_index < 2);
                    return Ok(Lvalue::Local {
                        frame,
                        local,
                        field: Some((field_index, field_ty)),
                    });
                },
            },
            // FIXME: do for globals what we did for locals
            Lvalue::Global(_) => self.force_allocation(base)?.to_ptr_and_extra(),
        };

        let offset = match base_extra {
            LvalueExtra::Vtable(tab) => {
                let (_, align) = self.size_and_align_of_dst(base_ty, Value::ByValPair(PrimVal::Ptr(base_ptr), PrimVal::Ptr(tab)))?;
                offset.abi_align(Align::from_bytes(align, align).unwrap()).bytes()
            }
            _ => offset.bytes(),
        };

        let ptr = base_ptr.offset(offset);

        let field_ty = self.monomorphize(field_ty, self.substs());

        if packed {
            let size = self.type_size(field_ty)?.expect("packed struct must be sized");
            self.memory.mark_packed(ptr, size);
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
                    ty::TyAdt(ref def, _) if def.is_box() => base_ty.boxed_ty(),
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
                assert!(n < len);
                let ptr = base_ptr.offset(n * elem_size);
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

                let ptr = base_ptr.offset(index * elem_size);
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
                let ptr = base_ptr.offset(u64::from(from) * elem_size);
                let extra = LvalueExtra::Length(n - u64::from(to) - u64::from(from));
                (ptr, extra)
            }
        };

        Ok(Lvalue::Ptr { ptr, extra })
    }

    pub(super) fn lvalue_ty(&self, lvalue: &mir::Lvalue<'tcx>) -> Ty<'tcx> {
        self.monomorphize(lvalue.ty(&self.mir(), self.tcx).to_ty(self.tcx), self.substs())
    }
}
