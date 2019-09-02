use crate::prelude::*;

fn codegen_field<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    base: Value,
    extra: Option<Value>,
    layout: TyLayout<'tcx>,
    field: mir::Field,
) -> (Value, TyLayout<'tcx>) {
    let field_offset = layout.fields.offset(field.index());
    let field_layout = layout.field(&*fx, field.index());

    let simple = |fx: &mut FunctionCx<_>| {
        if field_offset.bytes() > 0 {
            (
                fx.bcx.ins().iadd_imm(base, field_offset.bytes() as i64),
                field_layout,
            )
        } else {
            (base, field_layout)
        }
    };

    if let Some(extra) = extra {
        if !field_layout.is_unsized() {
            return simple(fx);
        }
        match field_layout.ty.sty {
            ty::Slice(..) | ty::Str | ty::Foreign(..) => return simple(fx),
            ty::Adt(def, _) if def.repr.packed() => {
                assert_eq!(layout.align.abi.bytes(), 1);
                return simple(fx);
            }
            _ => {
                // We have to align the offset for DST's
                let unaligned_offset = field_offset.bytes();
                let (_, unsized_align) = crate::unsize::size_and_align_of_dst(fx, field_layout.ty, extra);

                let one = fx.bcx.ins().iconst(pointer_ty(fx.tcx), 1);
                let align_sub_1 = fx.bcx.ins().isub(unsized_align, one);
                let and_lhs = fx.bcx.ins().iadd_imm(align_sub_1, unaligned_offset as i64);
                let zero = fx.bcx.ins().iconst(pointer_ty(fx.tcx), 0);
                let and_rhs = fx.bcx.ins().isub(zero, unsized_align);
                let offset = fx.bcx.ins().band(and_lhs, and_rhs);

                (
                    fx.bcx.ins().iadd(base, offset),
                    field_layout,
                )
            }
        }
    } else {
        simple(fx)
    }
}

fn scalar_pair_calculate_b_offset(tcx: TyCtxt<'_>, a_scalar: &Scalar, b_scalar: &Scalar) -> i32 {
    let b_offset = a_scalar
        .value
        .size(&tcx)
        .align_to(b_scalar.value.align(&tcx).abi);
    b_offset.bytes().try_into().unwrap()
}

/// A read-only value
#[derive(Debug, Copy, Clone)]
pub struct CValue<'tcx>(CValueInner, TyLayout<'tcx>);

#[derive(Debug, Copy, Clone)]
enum CValueInner {
    ByRef(Value),
    ByVal(Value),
    ByValPair(Value, Value),
}

impl<'tcx> CValue<'tcx> {
    pub fn by_ref(value: Value, layout: TyLayout<'tcx>) -> CValue<'tcx> {
        CValue(CValueInner::ByRef(value), layout)
    }

    pub fn by_val(value: Value, layout: TyLayout<'tcx>) -> CValue<'tcx> {
        CValue(CValueInner::ByVal(value), layout)
    }

    pub fn by_val_pair(value: Value, extra: Value, layout: TyLayout<'tcx>) -> CValue<'tcx> {
        CValue(CValueInner::ByValPair(value, extra), layout)
    }

    pub fn layout(&self) -> TyLayout<'tcx> {
        self.1
    }

    pub fn force_stack<'a>(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>) -> Value {
        let layout = self.1;
        match self.0 {
            CValueInner::ByRef(value) => value,
            CValueInner::ByVal(_) | CValueInner::ByValPair(_, _) => {
                let cplace = CPlace::new_stack_slot(fx, layout.ty);
                cplace.write_cvalue(fx, self);
                cplace.to_addr(fx)
            }
        }
    }

    pub fn try_to_addr(self) -> Option<Value> {
        match self.0 {
            CValueInner::ByRef(addr) => Some(addr),
            CValueInner::ByVal(_) | CValueInner::ByValPair(_, _) => None,
        }
    }

    /// Load a value with layout.abi of scalar
    pub fn load_scalar<'a>(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>) -> Value {
        let layout = self.1;
        match self.0 {
            CValueInner::ByRef(addr) => {
                let scalar = match layout.abi {
                    layout::Abi::Scalar(ref scalar) => scalar.clone(),
                    _ => unreachable!(),
                };
                let clif_ty = scalar_to_clif_type(fx.tcx, scalar);
                fx.bcx.ins().load(clif_ty, MemFlags::new(), addr, 0)
            }
            CValueInner::ByVal(value) => value,
            CValueInner::ByValPair(_, _) => bug!("Please use load_scalar_pair for ByValPair"),
        }
    }

    /// Load a value pair with layout.abi of scalar pair
    pub fn load_scalar_pair<'a>(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    ) -> (Value, Value) {
        let layout = self.1;
        match self.0 {
            CValueInner::ByRef(addr) => {
                let (a_scalar, b_scalar) = match &layout.abi {
                    layout::Abi::ScalarPair(a, b) => (a, b),
                    _ => unreachable!(),
                };
                let b_offset = scalar_pair_calculate_b_offset(fx.tcx, a_scalar, b_scalar);
                let clif_ty1 = scalar_to_clif_type(fx.tcx, a_scalar.clone());
                let clif_ty2 = scalar_to_clif_type(fx.tcx, b_scalar.clone());
                let val1 = fx.bcx.ins().load(clif_ty1, MemFlags::new(), addr, 0);
                let val2 = fx.bcx.ins().load(clif_ty2, MemFlags::new(), addr, b_offset);
                (val1, val2)
            }
            CValueInner::ByVal(_) => bug!("Please use load_scalar for ByVal"),
            CValueInner::ByValPair(val1, val2) => (val1, val2),
        }
    }

    pub fn value_field<'a>(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
        field: mir::Field,
    ) -> CValue<'tcx> {
        let layout = self.1;
        let base = match self.0 {
            CValueInner::ByRef(addr) => addr,
            _ => bug!("place_field for {:?}", self),
        };

        let (field_ptr, field_layout) = codegen_field(fx, base, None, layout, field);
        CValue::by_ref(field_ptr, field_layout)
    }

    pub fn unsize_value<'a>(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>, dest: CPlace<'tcx>) {
        crate::unsize::coerce_unsized_into(fx, self, dest);
    }

    /// If `ty` is signed, `const_val` must already be sign extended.
    pub fn const_val<'a>(
        fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
        ty: Ty<'tcx>,
        const_val: u128,
    ) -> CValue<'tcx> {
        let clif_ty = fx.clif_type(ty).unwrap();
        let layout = fx.layout_of(ty);

        let val = match ty.sty {
            ty::TyKind::Uint(UintTy::U128) | ty::TyKind::Int(IntTy::I128) => {
                let lsb = fx.bcx.ins().iconst(types::I64, const_val as u64 as i64);
                let msb = fx
                    .bcx
                    .ins()
                    .iconst(types::I64, (const_val >> 64) as u64 as i64);
                fx.bcx.ins().iconcat(lsb, msb)
            }
            ty::TyKind::Bool => {
                assert!(
                    const_val == 0 || const_val == 1,
                    "Invalid bool 0x{:032X}",
                    const_val
                );
                fx.bcx.ins().iconst(types::I8, const_val as i64)
            }
            ty::TyKind::Uint(_) | ty::TyKind::Ref(..) | ty::TyKind::RawPtr(..) => fx
                .bcx
                .ins()
                .iconst(clif_ty, u64::try_from(const_val).expect("uint") as i64),
            ty::TyKind::Int(_) => fx.bcx.ins().iconst(clif_ty, const_val as i128 as i64),
            _ => panic!(
                "CValue::const_val for non bool/integer/pointer type {:?} is not allowed",
                ty
            ),
        };

        CValue::by_val(val, layout)
    }

    pub fn unchecked_cast_to(self, layout: TyLayout<'tcx>) -> Self {
        CValue(self.0, layout)
    }
}

/// A place where you can write a value to or read a value from
#[derive(Debug, Copy, Clone)]
pub struct CPlace<'tcx> {
    inner: CPlaceInner,
    layout: TyLayout<'tcx>,
}

#[derive(Debug, Copy, Clone)]
pub enum CPlaceInner {
    Var(Local),
    Addr(Value, Option<Value>),
    Stack(StackSlot),
    NoPlace,
}

impl<'tcx> CPlace<'tcx> {
    pub fn layout(&self) -> TyLayout<'tcx> {
        self.layout
    }

    pub fn inner(&self) -> &CPlaceInner {
        &self.inner
    }

    pub fn no_place(layout: TyLayout<'tcx>) -> CPlace<'tcx> {
        CPlace {
            inner: CPlaceInner::NoPlace,
            layout,
        }
    }

    pub fn new_stack_slot(
        fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
        ty: Ty<'tcx>,
    ) -> CPlace<'tcx> {
        let layout = fx.layout_of(ty);
        assert!(!layout.is_unsized());
        if layout.size.bytes() == 0 {
            return CPlace {
                inner: CPlaceInner::NoPlace,
                layout,
            };
        }

        let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: layout.size.bytes() as u32,
            offset: None,
        });
        CPlace {
            inner: CPlaceInner::Stack(stack_slot),
            layout,
        }
    }

    pub fn new_var(
        fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
        local: Local,
        layout: TyLayout<'tcx>,
    ) -> CPlace<'tcx> {
        fx.bcx
            .declare_var(mir_var(local), fx.clif_type(layout.ty).unwrap());
        CPlace {
            inner: CPlaceInner::Var(local),
            layout,
        }
    }

    pub fn for_addr(addr: Value, layout: TyLayout<'tcx>) -> CPlace<'tcx> {
        CPlace {
            inner: CPlaceInner::Addr(addr, None),
            layout,
        }
    }

    pub fn for_addr_with_extra(addr: Value, extra: Value, layout: TyLayout<'tcx>) -> CPlace<'tcx> {
        CPlace {
            inner: CPlaceInner::Addr(addr, Some(extra)),
            layout,
        }
    }

    pub fn to_cvalue(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>) -> CValue<'tcx> {
        let layout = self.layout();
        match self.inner {
            CPlaceInner::Var(var) => CValue::by_val(fx.bcx.use_var(mir_var(var)), layout),
            CPlaceInner::Addr(addr, extra) => {
                assert!(extra.is_none(), "unsized values are not yet supported");
                CValue::by_ref(addr, layout)
            }
            CPlaceInner::Stack(stack_slot) => CValue::by_ref(
                fx.bcx.ins().stack_addr(fx.pointer_type, stack_slot, 0),
                layout,
            ),
            CPlaceInner::NoPlace => CValue::by_ref(
                fx.bcx
                    .ins()
                    .iconst(fx.pointer_type, fx.pointer_type.bytes() as i64),
                layout,
            ),
        }
    }

    pub fn to_addr(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>) -> Value {
        match self.to_addr_maybe_unsized(fx) {
            (addr, None) => addr,
            (_, Some(_)) => bug!("Expected sized cplace, found {:?}", self),
        }
    }

    pub fn to_addr_maybe_unsized(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    ) -> (Value, Option<Value>) {
        match self.inner {
            CPlaceInner::Addr(addr, extra) => (addr, extra),
            CPlaceInner::Stack(stack_slot) => (
                fx.bcx.ins().stack_addr(fx.pointer_type, stack_slot, 0),
                None,
            ),
            CPlaceInner::NoPlace => (fx.bcx.ins().iconst(fx.pointer_type, 45), None),
            CPlaceInner::Var(_) => bug!("Expected CPlace::Addr, found CPlace::Var"),
        }
    }

    pub fn write_cvalue(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>, from: CValue<'tcx>) {
        use rustc::hir::Mutability::*;

        let from_ty = from.layout().ty;
        let to_ty = self.layout().ty;

        fn assert_assignable<'tcx>(
            fx: &FunctionCx<'_, 'tcx, impl Backend>,
            from_ty: Ty<'tcx>,
            to_ty: Ty<'tcx>,
        ) {
            match (&from_ty.sty, &to_ty.sty) {
                (ty::Ref(_, t, MutImmutable), ty::Ref(_, u, MutImmutable))
                | (ty::Ref(_, t, MutMutable), ty::Ref(_, u, MutImmutable))
                | (ty::Ref(_, t, MutMutable), ty::Ref(_, u, MutMutable)) => {
                    assert_assignable(fx, t, u);
                    // &mut T -> &T is allowed
                    // &'a T -> &'b T is allowed
                }
                (ty::Ref(_, _, MutImmutable), ty::Ref(_, _, MutMutable)) => panic!(
                    "Cant assign value of type {} to place of type {}",
                    from_ty, to_ty
                ),
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

        assert_assignable(fx, from_ty, to_ty);

        let dst_layout = self.layout();
        let addr = match self.inner {
            CPlaceInner::Var(var) => {
                let data = from.load_scalar(fx);
                fx.bcx.def_var(mir_var(var), data);
                return;
            }
            CPlaceInner::Addr(addr, None) => addr,
            CPlaceInner::Stack(stack_slot) => {
                fx.bcx.ins().stack_addr(fx.pointer_type, stack_slot, 0)
            }
            CPlaceInner::NoPlace => {
                if dst_layout.abi != Abi::Uninhabited {
                    assert_eq!(dst_layout.size.bytes(), 0, "{:?}", dst_layout);
                }
                return;
            }
            CPlaceInner::Addr(_, Some(_)) => bug!("Can't write value to unsized place {:?}", self),
        };

        match from.0 {
            CValueInner::ByVal(val) => {
                fx.bcx.ins().store(MemFlags::new(), val, addr, 0);
            }
            CValueInner::ByValPair(value, extra) => match dst_layout.abi {
                Abi::ScalarPair(ref a_scalar, ref b_scalar) => {
                    let b_offset = scalar_pair_calculate_b_offset(fx.tcx, a_scalar, b_scalar);
                    fx.bcx.ins().store(MemFlags::new(), value, addr, 0);
                    fx.bcx.ins().store(MemFlags::new(), extra, addr, b_offset);
                }
                _ => bug!(
                    "Non ScalarPair abi {:?} for ByValPair CValue",
                    dst_layout.abi
                ),
            },
            CValueInner::ByRef(from_addr) => {
                let src_layout = from.1;
                let size = dst_layout.size.bytes();
                let src_align = src_layout.align.abi.bytes() as u8;
                let dst_align = dst_layout.align.abi.bytes() as u8;
                fx.bcx.emit_small_memcpy(
                    fx.module.target_config(),
                    addr,
                    from_addr,
                    size,
                    dst_align,
                    src_align,
                );
            }
        }
    }

    pub fn place_field(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
        field: mir::Field,
    ) -> CPlace<'tcx> {
        let layout = self.layout();
        let (base, extra) = self.to_addr_maybe_unsized(fx);

        let (field_ptr, field_layout) = codegen_field(fx, base, extra, layout, field);
        if field_layout.is_unsized() {
            CPlace::for_addr_with_extra(field_ptr, extra.unwrap(), field_layout)
        } else {
            CPlace::for_addr(field_ptr, field_layout)
        }
    }

    pub fn place_index(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
        index: Value,
    ) -> CPlace<'tcx> {
        let (elem_layout, addr) = match self.layout().ty.sty {
            ty::Array(elem_ty, _) => (fx.layout_of(elem_ty), self.to_addr(fx)),
            ty::Slice(elem_ty) => (fx.layout_of(elem_ty), self.to_addr_maybe_unsized(fx).0),
            _ => bug!("place_index({:?})", self.layout().ty),
        };

        let offset = fx
            .bcx
            .ins()
            .imul_imm(index, elem_layout.size.bytes() as i64);

        CPlace::for_addr(fx.bcx.ins().iadd(addr, offset), elem_layout)
    }

    pub fn place_deref(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>) -> CPlace<'tcx> {
        let inner_layout = fx.layout_of(self.layout().ty.builtin_deref(true).unwrap().ty);
        if !inner_layout.is_unsized() {
            CPlace::for_addr(self.to_cvalue(fx).load_scalar(fx), inner_layout)
        } else {
            let (addr, extra) = self.to_cvalue(fx).load_scalar_pair(fx);
            CPlace::for_addr_with_extra(addr, extra, inner_layout)
        }
    }

    pub fn write_place_ref(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>, dest: CPlace<'tcx>) {
        if !self.layout().is_unsized() {
            let ptr = CValue::by_val(self.to_addr(fx), dest.layout());
            dest.write_cvalue(fx, ptr);
        } else {
            let (value, extra) = self.to_addr_maybe_unsized(fx);
            let ptr = CValue::by_val_pair(
                value,
                extra.expect("unsized type without metadata"),
                dest.layout(),
            );
            dest.write_cvalue(fx, ptr);
        }
    }

    pub fn unchecked_cast_to(self, layout: TyLayout<'tcx>) -> Self {
        assert!(!self.layout().is_unsized());
        match self.inner {
            CPlaceInner::NoPlace => {
                assert!(layout.size.bytes() == 0);
            }
            _ => {}
        }
        CPlace {
            inner: self.inner,
            layout,
        }
    }

    pub fn downcast_variant(
        self,
        fx: &FunctionCx<'_, 'tcx, impl Backend>,
        variant: VariantIdx,
    ) -> Self {
        let layout = self.layout().for_variant(fx, variant);
        self.unchecked_cast_to(layout)
    }
}
