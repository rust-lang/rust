//! Definition of [`CValue`] and [`CPlace`]

use crate::prelude::*;

use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::immediates::Offset32;

fn codegen_field<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    base: Pointer,
    extra: Option<Value>,
    layout: TyAndLayout<'tcx>,
    field: mir::Field,
) -> (Pointer, TyAndLayout<'tcx>) {
    let field_offset = layout.fields.offset(field.index());
    let field_layout = layout.field(&*fx, field.index());

    let simple = |fx: &mut FunctionCx<'_, '_, _>| {
        (
            base.offset_i64(fx, i64::try_from(field_offset.bytes()).unwrap()),
            field_layout,
        )
    };

    if let Some(extra) = extra {
        if !field_layout.is_unsized() {
            return simple(fx);
        }
        match field_layout.ty.kind() {
            ty::Slice(..) | ty::Str | ty::Foreign(..) => simple(fx),
            ty::Adt(def, _) if def.repr.packed() => {
                assert_eq!(layout.align.abi.bytes(), 1);
                simple(fx)
            }
            _ => {
                // We have to align the offset for DST's
                let unaligned_offset = field_offset.bytes();
                let (_, unsized_align) =
                    crate::unsize::size_and_align_of_dst(fx, field_layout, extra);

                let one = fx.bcx.ins().iconst(pointer_ty(fx.tcx), 1);
                let align_sub_1 = fx.bcx.ins().isub(unsized_align, one);
                let and_lhs = fx.bcx.ins().iadd_imm(align_sub_1, unaligned_offset as i64);
                let zero = fx.bcx.ins().iconst(pointer_ty(fx.tcx), 0);
                let and_rhs = fx.bcx.ins().isub(zero, unsized_align);
                let offset = fx.bcx.ins().band(and_lhs, and_rhs);

                (base.offset_value(fx, offset), field_layout)
            }
        }
    } else {
        simple(fx)
    }
}

fn scalar_pair_calculate_b_offset(
    tcx: TyCtxt<'_>,
    a_scalar: &Scalar,
    b_scalar: &Scalar,
) -> Offset32 {
    let b_offset = a_scalar
        .value
        .size(&tcx)
        .align_to(b_scalar.value.align(&tcx).abi);
    Offset32::new(b_offset.bytes().try_into().unwrap())
}

/// A read-only value
#[derive(Debug, Copy, Clone)]
pub(crate) struct CValue<'tcx>(CValueInner, TyAndLayout<'tcx>);

#[derive(Debug, Copy, Clone)]
enum CValueInner {
    ByRef(Pointer, Option<Value>),
    ByVal(Value),
    ByValPair(Value, Value),
}

impl<'tcx> CValue<'tcx> {
    pub(crate) fn by_ref(ptr: Pointer, layout: TyAndLayout<'tcx>) -> CValue<'tcx> {
        CValue(CValueInner::ByRef(ptr, None), layout)
    }

    pub(crate) fn by_ref_unsized(
        ptr: Pointer,
        meta: Value,
        layout: TyAndLayout<'tcx>,
    ) -> CValue<'tcx> {
        CValue(CValueInner::ByRef(ptr, Some(meta)), layout)
    }

    pub(crate) fn by_val(value: Value, layout: TyAndLayout<'tcx>) -> CValue<'tcx> {
        CValue(CValueInner::ByVal(value), layout)
    }

    pub(crate) fn by_val_pair(
        value: Value,
        extra: Value,
        layout: TyAndLayout<'tcx>,
    ) -> CValue<'tcx> {
        CValue(CValueInner::ByValPair(value, extra), layout)
    }

    pub(crate) fn layout(&self) -> TyAndLayout<'tcx> {
        self.1
    }

    // FIXME remove
    pub(crate) fn force_stack(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    ) -> (Pointer, Option<Value>) {
        let layout = self.1;
        match self.0 {
            CValueInner::ByRef(ptr, meta) => (ptr, meta),
            CValueInner::ByVal(_) | CValueInner::ByValPair(_, _) => {
                let cplace = CPlace::new_stack_slot(fx, layout);
                cplace.write_cvalue(fx, self);
                (cplace.to_ptr(), None)
            }
        }
    }

    pub(crate) fn try_to_ptr(self) -> Option<(Pointer, Option<Value>)> {
        match self.0 {
            CValueInner::ByRef(ptr, meta) => Some((ptr, meta)),
            CValueInner::ByVal(_) | CValueInner::ByValPair(_, _) => None,
        }
    }

    /// Load a value with layout.abi of scalar
    pub(crate) fn load_scalar(self, fx: &mut FunctionCx<'_, 'tcx, impl Module>) -> Value {
        let layout = self.1;
        match self.0 {
            CValueInner::ByRef(ptr, None) => {
                let clif_ty = match layout.abi {
                    Abi::Scalar(ref scalar) => scalar_to_clif_type(fx.tcx, scalar.clone()),
                    Abi::Vector { ref element, count } => {
                        scalar_to_clif_type(fx.tcx, element.clone())
                            .by(u16::try_from(count).unwrap())
                            .unwrap()
                    }
                    _ => unreachable!("{:?}", layout.ty),
                };
                let mut flags = MemFlags::new();
                flags.set_notrap();
                ptr.load(fx, clif_ty, flags)
            }
            CValueInner::ByVal(value) => value,
            CValueInner::ByRef(_, Some(_)) => bug!("load_scalar for unsized value not allowed"),
            CValueInner::ByValPair(_, _) => bug!("Please use load_scalar_pair for ByValPair"),
        }
    }

    /// Load a value pair with layout.abi of scalar pair
    pub(crate) fn load_scalar_pair(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    ) -> (Value, Value) {
        let layout = self.1;
        match self.0 {
            CValueInner::ByRef(ptr, None) => {
                let (a_scalar, b_scalar) = match &layout.abi {
                    Abi::ScalarPair(a, b) => (a, b),
                    _ => unreachable!("load_scalar_pair({:?})", self),
                };
                let b_offset = scalar_pair_calculate_b_offset(fx.tcx, a_scalar, b_scalar);
                let clif_ty1 = scalar_to_clif_type(fx.tcx, a_scalar.clone());
                let clif_ty2 = scalar_to_clif_type(fx.tcx, b_scalar.clone());
                let mut flags = MemFlags::new();
                flags.set_notrap();
                let val1 = ptr.load(fx, clif_ty1, flags);
                let val2 = ptr.offset(fx, b_offset).load(fx, clif_ty2, flags);
                (val1, val2)
            }
            CValueInner::ByRef(_, Some(_)) => {
                bug!("load_scalar_pair for unsized value not allowed")
            }
            CValueInner::ByVal(_) => bug!("Please use load_scalar for ByVal"),
            CValueInner::ByValPair(val1, val2) => (val1, val2),
        }
    }

    pub(crate) fn value_field(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
        field: mir::Field,
    ) -> CValue<'tcx> {
        let layout = self.1;
        match self.0 {
            CValueInner::ByVal(val) => match layout.abi {
                Abi::Vector { element: _, count } => {
                    let count = u8::try_from(count).expect("SIMD type with more than 255 lanes???");
                    let field = u8::try_from(field.index()).unwrap();
                    assert!(field < count);
                    let lane = fx.bcx.ins().extractlane(val, field);
                    let field_layout = layout.field(&*fx, usize::from(field));
                    CValue::by_val(lane, field_layout)
                }
                _ => unreachable!("value_field for ByVal with abi {:?}", layout.abi),
            },
            CValueInner::ByValPair(val1, val2) => match layout.abi {
                Abi::ScalarPair(_, _) => {
                    let val = match field.as_u32() {
                        0 => val1,
                        1 => val2,
                        _ => bug!("field should be 0 or 1"),
                    };
                    let field_layout = layout.field(&*fx, usize::from(field));
                    CValue::by_val(val, field_layout)
                }
                _ => unreachable!("value_field for ByValPair with abi {:?}", layout.abi),
            },
            CValueInner::ByRef(ptr, None) => {
                let (field_ptr, field_layout) = codegen_field(fx, ptr, None, layout, field);
                CValue::by_ref(field_ptr, field_layout)
            }
            CValueInner::ByRef(_, Some(_)) => todo!(),
        }
    }

    pub(crate) fn unsize_value(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
        dest: CPlace<'tcx>,
    ) {
        crate::unsize::coerce_unsized_into(fx, self, dest);
    }

    /// If `ty` is signed, `const_val` must already be sign extended.
    pub(crate) fn const_val(
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
        layout: TyAndLayout<'tcx>,
        const_val: ty::ScalarInt,
    ) -> CValue<'tcx> {
        assert_eq!(const_val.size(), layout.size);
        use cranelift_codegen::ir::immediates::{Ieee32, Ieee64};

        let clif_ty = fx.clif_type(layout.ty).unwrap();

        if let ty::Bool = layout.ty.kind() {
            assert!(
                const_val == ty::ScalarInt::FALSE || const_val == ty::ScalarInt::TRUE,
                "Invalid bool 0x{:032X}",
                const_val
            );
        }

        let val = match layout.ty.kind() {
            ty::Uint(UintTy::U128) | ty::Int(IntTy::I128) => {
                let const_val = const_val.to_bits(layout.size).unwrap();
                let lsb = fx.bcx.ins().iconst(types::I64, const_val as u64 as i64);
                let msb = fx
                    .bcx
                    .ins()
                    .iconst(types::I64, (const_val >> 64) as u64 as i64);
                fx.bcx.ins().iconcat(lsb, msb)
            }
            ty::Bool | ty::Char | ty::Uint(_) | ty::Int(_) | ty::Ref(..)
            | ty::RawPtr(..) => {
                fx
                    .bcx
                    .ins()
                    .iconst(clif_ty, const_val.to_bits(layout.size).unwrap() as i64)
            }
            ty::Float(FloatTy::F32) => {
                fx.bcx.ins().f32const(Ieee32::with_bits(u32::try_from(const_val).unwrap()))
            }
            ty::Float(FloatTy::F64) => {
                fx.bcx.ins().f64const(Ieee64::with_bits(u64::try_from(const_val).unwrap()))
            }
            _ => panic!(
                "CValue::const_val for non bool/char/float/integer/pointer type {:?} is not allowed",
                layout.ty
            ),
        };

        CValue::by_val(val, layout)
    }

    pub(crate) fn cast_pointer_to(self, layout: TyAndLayout<'tcx>) -> Self {
        assert!(matches!(
            self.layout().ty.kind(),
            ty::Ref(..) | ty::RawPtr(..) | ty::FnPtr(..)
        ));
        assert!(matches!(
            layout.ty.kind(),
            ty::Ref(..) | ty::RawPtr(..) | ty::FnPtr(..)
        ));
        assert_eq!(self.layout().abi, layout.abi);
        CValue(self.0, layout)
    }
}

/// A place where you can write a value to or read a value from
#[derive(Debug, Copy, Clone)]
pub(crate) struct CPlace<'tcx> {
    inner: CPlaceInner,
    layout: TyAndLayout<'tcx>,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum CPlaceInner {
    Var(Local, Variable),
    VarPair(Local, Variable, Variable),
    VarLane(Local, Variable, u8),
    Addr(Pointer, Option<Value>),
}

impl<'tcx> CPlace<'tcx> {
    pub(crate) fn layout(&self) -> TyAndLayout<'tcx> {
        self.layout
    }

    pub(crate) fn inner(&self) -> &CPlaceInner {
        &self.inner
    }

    pub(crate) fn no_place(layout: TyAndLayout<'tcx>) -> CPlace<'tcx> {
        CPlace {
            inner: CPlaceInner::Addr(Pointer::dangling(layout.align.pref), None),
            layout,
        }
    }

    pub(crate) fn new_stack_slot(
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
        layout: TyAndLayout<'tcx>,
    ) -> CPlace<'tcx> {
        assert!(!layout.is_unsized());
        if layout.size.bytes() == 0 {
            return CPlace::no_place(layout);
        }

        let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: u32::try_from(layout.size.bytes()).unwrap(),
            offset: None,
        });
        CPlace {
            inner: CPlaceInner::Addr(Pointer::stack_slot(stack_slot), None),
            layout,
        }
    }

    pub(crate) fn new_var(
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
        local: Local,
        layout: TyAndLayout<'tcx>,
    ) -> CPlace<'tcx> {
        let var = Variable::with_u32(fx.next_ssa_var);
        fx.next_ssa_var += 1;
        fx.bcx.declare_var(var, fx.clif_type(layout.ty).unwrap());
        CPlace {
            inner: CPlaceInner::Var(local, var),
            layout,
        }
    }

    pub(crate) fn new_var_pair(
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
        local: Local,
        layout: TyAndLayout<'tcx>,
    ) -> CPlace<'tcx> {
        let var1 = Variable::with_u32(fx.next_ssa_var);
        fx.next_ssa_var += 1;
        let var2 = Variable::with_u32(fx.next_ssa_var);
        fx.next_ssa_var += 1;

        let (ty1, ty2) = fx.clif_pair_type(layout.ty).unwrap();
        fx.bcx.declare_var(var1, ty1);
        fx.bcx.declare_var(var2, ty2);
        CPlace {
            inner: CPlaceInner::VarPair(local, var1, var2),
            layout,
        }
    }

    pub(crate) fn for_ptr(ptr: Pointer, layout: TyAndLayout<'tcx>) -> CPlace<'tcx> {
        CPlace {
            inner: CPlaceInner::Addr(ptr, None),
            layout,
        }
    }

    pub(crate) fn for_ptr_with_extra(
        ptr: Pointer,
        extra: Value,
        layout: TyAndLayout<'tcx>,
    ) -> CPlace<'tcx> {
        CPlace {
            inner: CPlaceInner::Addr(ptr, Some(extra)),
            layout,
        }
    }

    pub(crate) fn to_cvalue(self, fx: &mut FunctionCx<'_, 'tcx, impl Module>) -> CValue<'tcx> {
        let layout = self.layout();
        match self.inner {
            CPlaceInner::Var(_local, var) => {
                let val = fx.bcx.use_var(var);
                fx.bcx
                    .set_val_label(val, cranelift_codegen::ir::ValueLabel::new(var.index()));
                CValue::by_val(val, layout)
            }
            CPlaceInner::VarPair(_local, var1, var2) => {
                let val1 = fx.bcx.use_var(var1);
                fx.bcx
                    .set_val_label(val1, cranelift_codegen::ir::ValueLabel::new(var1.index()));
                let val2 = fx.bcx.use_var(var2);
                fx.bcx
                    .set_val_label(val2, cranelift_codegen::ir::ValueLabel::new(var2.index()));
                CValue::by_val_pair(val1, val2, layout)
            }
            CPlaceInner::VarLane(_local, var, lane) => {
                let val = fx.bcx.use_var(var);
                fx.bcx
                    .set_val_label(val, cranelift_codegen::ir::ValueLabel::new(var.index()));
                let val = fx.bcx.ins().extractlane(val, lane);
                CValue::by_val(val, layout)
            }
            CPlaceInner::Addr(ptr, extra) => {
                if let Some(extra) = extra {
                    CValue::by_ref_unsized(ptr, extra, layout)
                } else {
                    CValue::by_ref(ptr, layout)
                }
            }
        }
    }

    pub(crate) fn to_ptr(self) -> Pointer {
        match self.to_ptr_maybe_unsized() {
            (ptr, None) => ptr,
            (_, Some(_)) => bug!("Expected sized cplace, found {:?}", self),
        }
    }

    pub(crate) fn to_ptr_maybe_unsized(self) -> (Pointer, Option<Value>) {
        match self.inner {
            CPlaceInner::Addr(ptr, extra) => (ptr, extra),
            CPlaceInner::Var(_, _)
            | CPlaceInner::VarPair(_, _, _)
            | CPlaceInner::VarLane(_, _, _) => bug!("Expected CPlace::Addr, found {:?}", self),
        }
    }

    pub(crate) fn write_cvalue(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
        from: CValue<'tcx>,
    ) {
        fn assert_assignable<'tcx>(
            fx: &FunctionCx<'_, 'tcx, impl Module>,
            from_ty: Ty<'tcx>,
            to_ty: Ty<'tcx>,
        ) {
            match (&from_ty.kind(), &to_ty.kind()) {
                (ty::Ref(_, a, _), ty::Ref(_, b, _))
                | (
                    ty::RawPtr(TypeAndMut { ty: a, mutbl: _ }),
                    ty::RawPtr(TypeAndMut { ty: b, mutbl: _ }),
                ) => {
                    assert_assignable(fx, a, b);
                }
                (ty::FnPtr(_), ty::FnPtr(_)) => {
                    let from_sig = fx.tcx.normalize_erasing_late_bound_regions(
                        ParamEnv::reveal_all(),
                        &from_ty.fn_sig(fx.tcx),
                    );
                    let to_sig = fx.tcx.normalize_erasing_late_bound_regions(
                        ParamEnv::reveal_all(),
                        &to_ty.fn_sig(fx.tcx),
                    );
                    assert_eq!(
                        from_sig, to_sig,
                        "Can't write fn ptr with incompatible sig {:?} to place with sig {:?}\n\n{:#?}",
                        from_sig, to_sig, fx,
                    );
                    // fn(&T) -> for<'l> fn(&'l T) is allowed
                }
                (ty::Dynamic(from_traits, _), ty::Dynamic(to_traits, _)) => {
                    let from_traits = fx
                        .tcx
                        .normalize_erasing_late_bound_regions(ParamEnv::reveal_all(), from_traits);
                    let to_traits = fx
                        .tcx
                        .normalize_erasing_late_bound_regions(ParamEnv::reveal_all(), to_traits);
                    assert_eq!(
                        from_traits, to_traits,
                        "Can't write trait object of incompatible traits {:?} to place with traits {:?}\n\n{:#?}",
                        from_traits, to_traits, fx,
                    );
                    // dyn for<'r> Trait<'r> -> dyn Trait<'_> is allowed
                }
                _ => {
                    assert_eq!(
                        from_ty,
                        to_ty,
                        "Can't write value with incompatible type {:?} to place with type {:?}\n\n{:#?}",
                        from_ty,
                        to_ty,
                        fx,
                    );
                }
            }
        }

        assert_assignable(fx, from.layout().ty, self.layout().ty);

        self.write_cvalue_maybe_transmute(fx, from, "write_cvalue");
    }

    pub(crate) fn write_cvalue_transmute(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
        from: CValue<'tcx>,
    ) {
        self.write_cvalue_maybe_transmute(fx, from, "write_cvalue_transmute");
    }

    fn write_cvalue_maybe_transmute(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
        from: CValue<'tcx>,
        #[cfg_attr(not(debug_assertions), allow(unused_variables))] method: &'static str,
    ) {
        fn transmute_value<'tcx>(
            fx: &mut FunctionCx<'_, 'tcx, impl Module>,
            var: Variable,
            data: Value,
            dst_ty: Type,
        ) {
            let src_ty = fx.bcx.func.dfg.value_type(data);
            assert_eq!(
                src_ty.bytes(),
                dst_ty.bytes(),
                "write_cvalue_transmute: {:?} -> {:?}",
                src_ty,
                dst_ty,
            );
            let data = match (src_ty, dst_ty) {
                (_, _) if src_ty == dst_ty => data,

                // This is a `write_cvalue_transmute`.
                (types::I32, types::F32)
                | (types::F32, types::I32)
                | (types::I64, types::F64)
                | (types::F64, types::I64) => fx.bcx.ins().bitcast(dst_ty, data),
                _ if src_ty.is_vector() && dst_ty.is_vector() => {
                    fx.bcx.ins().raw_bitcast(dst_ty, data)
                }
                _ if src_ty.is_vector() || dst_ty.is_vector() => {
                    // FIXME do something more efficient for transmutes between vectors and integers.
                    let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size: src_ty.bytes(),
                        offset: None,
                    });
                    let ptr = Pointer::stack_slot(stack_slot);
                    ptr.store(fx, data, MemFlags::trusted());
                    ptr.load(fx, dst_ty, MemFlags::trusted())
                }
                _ => unreachable!("write_cvalue_transmute: {:?} -> {:?}", src_ty, dst_ty),
            };
            fx.bcx
                .set_val_label(data, cranelift_codegen::ir::ValueLabel::new(var.index()));
            fx.bcx.def_var(var, data);
        }

        assert_eq!(self.layout().size, from.layout().size);

        #[cfg(debug_assertions)]
        {
            use cranelift_codegen::cursor::{Cursor, CursorPosition};
            let cur_block = match fx.bcx.cursor().position() {
                CursorPosition::After(block) => block,
                _ => unreachable!(),
            };
            fx.add_comment(
                fx.bcx.func.layout.last_inst(cur_block).unwrap(),
                format!(
                    "{}: {:?}: {:?} <- {:?}: {:?}",
                    method,
                    self.inner(),
                    self.layout().ty,
                    from.0,
                    from.layout().ty
                ),
            );
        }

        let dst_layout = self.layout();
        let to_ptr = match self.inner {
            CPlaceInner::Var(_local, var) => {
                let data = CValue(from.0, dst_layout).load_scalar(fx);
                let dst_ty = fx.clif_type(self.layout().ty).unwrap();
                transmute_value(fx, var, data, dst_ty);
                return;
            }
            CPlaceInner::VarPair(_local, var1, var2) => {
                let (data1, data2) = CValue(from.0, dst_layout).load_scalar_pair(fx);
                let (dst_ty1, dst_ty2) = fx.clif_pair_type(self.layout().ty).unwrap();
                transmute_value(fx, var1, data1, dst_ty1);
                transmute_value(fx, var2, data2, dst_ty2);
                return;
            }
            CPlaceInner::VarLane(_local, var, lane) => {
                let data = from.load_scalar(fx);

                // First get the old vector
                let vector = fx.bcx.use_var(var);
                fx.bcx
                    .set_val_label(vector, cranelift_codegen::ir::ValueLabel::new(var.index()));

                // Next insert the written lane into the vector
                let vector = fx.bcx.ins().insertlane(vector, data, lane);

                // Finally write the new vector
                fx.bcx
                    .set_val_label(vector, cranelift_codegen::ir::ValueLabel::new(var.index()));
                fx.bcx.def_var(var, vector);

                return;
            }
            CPlaceInner::Addr(ptr, None) => {
                if dst_layout.size == Size::ZERO || dst_layout.abi == Abi::Uninhabited {
                    return;
                }
                ptr
            }
            CPlaceInner::Addr(_, Some(_)) => bug!("Can't write value to unsized place {:?}", self),
        };

        let mut flags = MemFlags::new();
        flags.set_notrap();
        match from.layout().abi {
            // FIXME make Abi::Vector work too
            Abi::Scalar(_) => {
                let val = from.load_scalar(fx);
                to_ptr.store(fx, val, flags);
                return;
            }
            Abi::ScalarPair(ref a_scalar, ref b_scalar) => {
                let (value, extra) = from.load_scalar_pair(fx);
                let b_offset = scalar_pair_calculate_b_offset(fx.tcx, a_scalar, b_scalar);
                to_ptr.store(fx, value, flags);
                to_ptr.offset(fx, b_offset).store(fx, extra, flags);
                return;
            }
            _ => {}
        }

        match from.0 {
            CValueInner::ByVal(val) => {
                to_ptr.store(fx, val, flags);
            }
            CValueInner::ByValPair(_, _) => {
                bug!(
                    "Non ScalarPair abi {:?} for ByValPair CValue",
                    dst_layout.abi
                );
            }
            CValueInner::ByRef(from_ptr, None) => {
                let from_addr = from_ptr.get_addr(fx);
                let to_addr = to_ptr.get_addr(fx);
                let src_layout = from.1;
                let size = dst_layout.size.bytes();
                let src_align = src_layout.align.abi.bytes() as u8;
                let dst_align = dst_layout.align.abi.bytes() as u8;
                fx.bcx.emit_small_memory_copy(
                    fx.cx.module.target_config(),
                    to_addr,
                    from_addr,
                    size,
                    dst_align,
                    src_align,
                    true,
                );
            }
            CValueInner::ByRef(_, Some(_)) => todo!(),
        }
    }

    pub(crate) fn place_field(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
        field: mir::Field,
    ) -> CPlace<'tcx> {
        let layout = self.layout();

        match self.inner {
            CPlaceInner::Var(local, var) => {
                if let Abi::Vector { .. } = layout.abi {
                    return CPlace {
                        inner: CPlaceInner::VarLane(local, var, field.as_u32().try_into().unwrap()),
                        layout: layout.field(fx, field.as_u32().try_into().unwrap()),
                    };
                }
            }
            CPlaceInner::VarPair(local, var1, var2) => {
                let layout = layout.field(&*fx, field.index());

                match field.as_u32() {
                    0 => {
                        return CPlace {
                            inner: CPlaceInner::Var(local, var1),
                            layout,
                        }
                    }
                    1 => {
                        return CPlace {
                            inner: CPlaceInner::Var(local, var2),
                            layout,
                        }
                    }
                    _ => unreachable!("field should be 0 or 1"),
                }
            }
            _ => {}
        }

        let (base, extra) = self.to_ptr_maybe_unsized();

        let (field_ptr, field_layout) = codegen_field(fx, base, extra, layout, field);
        if field_layout.is_unsized() {
            CPlace::for_ptr_with_extra(field_ptr, extra.unwrap(), field_layout)
        } else {
            CPlace::for_ptr(field_ptr, field_layout)
        }
    }

    pub(crate) fn place_index(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
        index: Value,
    ) -> CPlace<'tcx> {
        let (elem_layout, ptr) = match self.layout().ty.kind() {
            ty::Array(elem_ty, _) => (fx.layout_of(elem_ty), self.to_ptr()),
            ty::Slice(elem_ty) => (fx.layout_of(elem_ty), self.to_ptr_maybe_unsized().0),
            _ => bug!("place_index({:?})", self.layout().ty),
        };

        let offset = fx
            .bcx
            .ins()
            .imul_imm(index, elem_layout.size.bytes() as i64);

        CPlace::for_ptr(ptr.offset_value(fx, offset), elem_layout)
    }

    pub(crate) fn place_deref(self, fx: &mut FunctionCx<'_, 'tcx, impl Module>) -> CPlace<'tcx> {
        let inner_layout = fx.layout_of(self.layout().ty.builtin_deref(true).unwrap().ty);
        if has_ptr_meta(fx.tcx, inner_layout.ty) {
            let (addr, extra) = self.to_cvalue(fx).load_scalar_pair(fx);
            CPlace::for_ptr_with_extra(Pointer::new(addr), extra, inner_layout)
        } else {
            CPlace::for_ptr(
                Pointer::new(self.to_cvalue(fx).load_scalar(fx)),
                inner_layout,
            )
        }
    }

    pub(crate) fn place_ref(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Module>,
        layout: TyAndLayout<'tcx>,
    ) -> CValue<'tcx> {
        if has_ptr_meta(fx.tcx, self.layout().ty) {
            let (ptr, extra) = self.to_ptr_maybe_unsized();
            CValue::by_val_pair(
                ptr.get_addr(fx),
                extra.expect("unsized type without metadata"),
                layout,
            )
        } else {
            CValue::by_val(self.to_ptr().get_addr(fx), layout)
        }
    }

    pub(crate) fn downcast_variant(
        self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        variant: VariantIdx,
    ) -> Self {
        assert!(!self.layout().is_unsized());
        let layout = self.layout().for_variant(fx, variant);
        CPlace {
            inner: self.inner,
            layout,
        }
    }
}
