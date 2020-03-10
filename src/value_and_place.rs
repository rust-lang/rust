use crate::prelude::*;

use cranelift_codegen::ir::immediates::Offset32;

fn codegen_field<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    base: Pointer,
    extra: Option<Value>,
    layout: TyLayout<'tcx>,
    field: mir::Field,
) -> (Pointer, TyLayout<'tcx>) {
    let field_offset = layout.fields.offset(field.index());
    let field_layout = layout.field(&*fx, field.index());

    let simple = |fx: &mut FunctionCx<_>| {
        (
            base.offset_i64(fx, i64::try_from(field_offset.bytes()).unwrap()),
            field_layout,
        )
    };

    if let Some(extra) = extra {
        if !field_layout.is_unsized() {
            return simple(fx);
        }
        match field_layout.ty.kind {
            ty::Slice(..) | ty::Str | ty::Foreign(..) => return simple(fx),
            ty::Adt(def, _) if def.repr.packed() => {
                assert_eq!(layout.align.abi.bytes(), 1);
                return simple(fx);
            }
            _ => {
                // We have to align the offset for DST's
                let unaligned_offset = field_offset.bytes();
                let (_, unsized_align) = crate::unsize::size_and_align_of_dst(fx, field_layout, extra);

                let one = fx.bcx.ins().iconst(pointer_ty(fx.tcx), 1);
                let align_sub_1 = fx.bcx.ins().isub(unsized_align, one);
                let and_lhs = fx.bcx.ins().iadd_imm(align_sub_1, unaligned_offset as i64);
                let zero = fx.bcx.ins().iconst(pointer_ty(fx.tcx), 0);
                let and_rhs = fx.bcx.ins().isub(zero, unsized_align);
                let offset = fx.bcx.ins().band(and_lhs, and_rhs);

                (
                    base.offset_value(fx, offset),
                    field_layout,
                )
            }
        }
    } else {
        simple(fx)
    }
}

fn scalar_pair_calculate_b_offset(tcx: TyCtxt<'_>, a_scalar: &Scalar, b_scalar: &Scalar) -> Offset32 {
    let b_offset = a_scalar
        .value
        .size(&tcx)
        .align_to(b_scalar.value.align(&tcx).abi);
    Offset32::new(b_offset.bytes().try_into().unwrap())
}

/// A read-only value
#[derive(Debug, Copy, Clone)]
pub struct CValue<'tcx>(CValueInner, TyLayout<'tcx>);

#[derive(Debug, Copy, Clone)]
enum CValueInner {
    ByRef(Pointer, Option<Value>),
    ByVal(Value),
    ByValPair(Value, Value),
}

impl<'tcx> CValue<'tcx> {
    pub fn by_ref(ptr: Pointer, layout: TyLayout<'tcx>) -> CValue<'tcx> {
        CValue(CValueInner::ByRef(ptr, None), layout)
    }

    pub fn by_ref_unsized(ptr: Pointer, meta: Value, layout: TyLayout<'tcx>) -> CValue<'tcx> {
        CValue(CValueInner::ByRef(ptr, Some(meta)), layout)
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

    // FIXME remove
    pub fn force_stack<'a>(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>) -> (Pointer, Option<Value>) {
        let layout = self.1;
        match self.0 {
            CValueInner::ByRef(ptr, meta) => (ptr, meta),
            CValueInner::ByVal(_) | CValueInner::ByValPair(_, _) => {
                let cplace = CPlace::new_stack_slot(fx, layout);
                cplace.write_cvalue(fx, self);
                (cplace.to_ptr(fx), None)
            }
        }
    }

    pub fn try_to_ptr(self) -> Option<(Pointer, Option<Value>)> {
        match self.0 {
            CValueInner::ByRef(ptr, meta) => Some((ptr, meta)),
            CValueInner::ByVal(_) | CValueInner::ByValPair(_, _) => None,
        }
    }

    /// Load a value with layout.abi of scalar
    pub fn load_scalar<'a>(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>) -> Value {
        let layout = self.1;
        match self.0 {
            CValueInner::ByRef(ptr, None) => {
                let clif_ty = match layout.abi {
                    layout::Abi::Scalar(ref scalar) => scalar_to_clif_type(fx.tcx, scalar.clone()),
                    layout::Abi::Vector { ref element, count } => {
                        scalar_to_clif_type(fx.tcx, element.clone())
                            .by(u16::try_from(count).unwrap()).unwrap()
                    }
                    _ => unreachable!(),
                };
                ptr.load(fx, clif_ty, MemFlags::new())
            }
            CValueInner::ByVal(value) => value,
            CValueInner::ByRef(_, Some(_)) => bug!("load_scalar for unsized value not allowed"),
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
            CValueInner::ByRef(ptr, None) => {
                let (a_scalar, b_scalar) = match &layout.abi {
                    layout::Abi::ScalarPair(a, b) => (a, b),
                    _ => unreachable!("load_scalar_pair({:?})", self),
                };
                let b_offset = scalar_pair_calculate_b_offset(fx.tcx, a_scalar, b_scalar);
                let clif_ty1 = scalar_to_clif_type(fx.tcx, a_scalar.clone());
                let clif_ty2 = scalar_to_clif_type(fx.tcx, b_scalar.clone());
                let val1 = ptr.load(fx, clif_ty1, MemFlags::new());
                let val2 = ptr.offset(fx, b_offset).load(fx, clif_ty2, MemFlags::new());
                (val1, val2)
            }
            CValueInner::ByRef(_, Some(_)) => bug!("load_scalar_pair for unsized value not allowed"),
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
        match self.0 {
            CValueInner::ByVal(val) => {
                match layout.abi {
                    layout::Abi::Vector { element: _, count } => {
                        let count = u8::try_from(count).expect("SIMD type with more than 255 lanes???");
                        let field = u8::try_from(field.index()).unwrap();
                        assert!(field < count);
                        let lane = fx.bcx.ins().extractlane(val, field);
                        let field_layout = layout.field(&*fx, usize::from(field));
                        CValue::by_val(lane, field_layout)
                    }
                    _ => unreachable!("value_field for ByVal with abi {:?}", layout.abi),
                }
            }
            CValueInner::ByRef(ptr, None) => {
                let (field_ptr, field_layout) = codegen_field(fx, ptr, None, layout, field);
                CValue::by_ref(field_ptr, field_layout)
            }
            CValueInner::ByRef(_, Some(_)) => todo!(),
            _ => bug!("place_field for {:?}", self),
        }
    }

    pub fn unsize_value<'a>(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>, dest: CPlace<'tcx>) {
        crate::unsize::coerce_unsized_into(fx, self, dest);
    }

    /// If `ty` is signed, `const_val` must already be sign extended.
    pub fn const_val(
        fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
        layout: TyLayout<'tcx>,
        const_val: u128,
    ) -> CValue<'tcx> {
        let clif_ty = fx.clif_type(layout.ty).unwrap();

        match layout.ty.kind {
            ty::TyKind::Bool => {
                assert!(const_val == 0 || const_val == 1, "Invalid bool 0x{:032X}", const_val);
            }
            _ => {}
        }

        let val = match layout.ty.kind {
            ty::TyKind::Uint(UintTy::U128) | ty::TyKind::Int(IntTy::I128) => {
                let lsb = fx.bcx.ins().iconst(types::I64, const_val as u64 as i64);
                let msb = fx
                    .bcx
                    .ins()
                    .iconst(types::I64, (const_val >> 64) as u64 as i64);
                fx.bcx.ins().iconcat(lsb, msb)
            }
            ty::TyKind::Bool | ty::TyKind::Char | ty::TyKind::Uint(_) | ty::TyKind::Ref(..)
            | ty::TyKind::RawPtr(..) => {
                fx
                    .bcx
                    .ins()
                    .iconst(clif_ty, u64::try_from(const_val).expect("uint") as i64)
            }
            ty::TyKind::Int(_) => {
                let const_val = rustc::mir::interpret::sign_extend(const_val, layout.size);
                fx.bcx.ins().iconst(clif_ty, i64::try_from(const_val as i128).unwrap())
            }
            ty::TyKind::Float(FloatTy::F32) => {
                fx.bcx.ins().f32const(Ieee32::with_bits(u32::try_from(const_val).unwrap()))
            }
            ty::TyKind::Float(FloatTy::F64) => {
                fx.bcx.ins().f64const(Ieee64::with_bits(u64::try_from(const_val).unwrap()))
            }
            _ => panic!(
                "CValue::const_val for non bool/char/float/integer/pointer type {:?} is not allowed",
                layout.ty
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
    Addr(Pointer, Option<Value>),
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
        layout: TyLayout<'tcx>,
    ) -> CPlace<'tcx> {
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
            inner: CPlaceInner::Addr(Pointer::stack_slot(stack_slot), None),
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

    pub fn for_ptr(ptr: Pointer, layout: TyLayout<'tcx>) -> CPlace<'tcx> {
        CPlace {
            inner: CPlaceInner::Addr(ptr, None),
            layout,
        }
    }

    pub fn for_ptr_with_extra(ptr: Pointer, extra: Value, layout: TyLayout<'tcx>) -> CPlace<'tcx> {
        CPlace {
            inner: CPlaceInner::Addr(ptr, Some(extra)),
            layout,
        }
    }

    pub fn to_cvalue(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>) -> CValue<'tcx> {
        let layout = self.layout();
        match self.inner {
            CPlaceInner::Var(var) => {
                let val = fx.bcx.use_var(mir_var(var));
                fx.bcx.set_val_label(val, cranelift_codegen::ir::ValueLabel::from_u32(var.as_u32()));
                CValue::by_val(val, layout)
            }
            CPlaceInner::Addr(ptr, extra) => {
                if let Some(extra) = extra {
                    CValue::by_ref_unsized(ptr, extra, layout)
                } else {
                    CValue::by_ref(ptr, layout)
                }
            }
            CPlaceInner::NoPlace => CValue::by_ref(
                Pointer::const_addr(fx, i64::try_from(self.layout.align.pref.bytes()).unwrap()),
                layout,
            ),
        }
    }

    pub fn to_ptr(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>) -> Pointer {
        match self.to_ptr_maybe_unsized(fx) {
            (ptr, None) => ptr,
            (_, Some(_)) => bug!("Expected sized cplace, found {:?}", self),
        }
    }

    pub fn to_ptr_maybe_unsized(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    ) -> (Pointer, Option<Value>) {
        match self.inner {
            CPlaceInner::Addr(ptr, extra) => (ptr, extra),
            CPlaceInner::NoPlace => {
                (
                    Pointer::const_addr(fx, i64::try_from(self.layout.align.pref.bytes()).unwrap()),
                    None,
                )
            }
            CPlaceInner::Var(_) => bug!("Expected CPlace::Addr, found CPlace::Var"),
        }
    }

    pub fn write_cvalue(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>, from: CValue<'tcx>) {
        #[cfg(debug_assertions)]
        {
            use cranelift_codegen::cursor::{Cursor, CursorPosition};
            let cur_block = match fx.bcx.cursor().position() {
                CursorPosition::After(block) => block,
                _ => unreachable!(),
            };
            fx.add_comment(
                fx.bcx.func.layout.last_inst(cur_block).unwrap(),
                format!("write_cvalue: {:?} <- {:?}",self, from),
            );
        }

        let from_ty = from.layout().ty;
        let to_ty = self.layout().ty;

        fn assert_assignable<'tcx>(
            fx: &FunctionCx<'_, 'tcx, impl Backend>,
            from_ty: Ty<'tcx>,
            to_ty: Ty<'tcx>,
        ) {
            match (&from_ty.kind, &to_ty.kind) {
                (ty::Ref(_, t, Mutability::Not), ty::Ref(_, u, Mutability::Not))
                | (ty::Ref(_, t, Mutability::Mut), ty::Ref(_, u, Mutability::Not))
                | (ty::Ref(_, t, Mutability::Mut), ty::Ref(_, u, Mutability::Mut)) => {
                    assert_assignable(fx, t, u);
                    // &mut T -> &T is allowed
                    // &'a T -> &'b T is allowed
                }
                (ty::Ref(_, _, Mutability::Not), ty::Ref(_, _, Mutability::Mut)) => panic!(
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
        let to_ptr = match self.inner {
            CPlaceInner::Var(var) => {
                let data = from.load_scalar(fx);
                fx.bcx.set_val_label(data, cranelift_codegen::ir::ValueLabel::from_u32(var.as_u32()));
                fx.bcx.def_var(mir_var(var), data);
                return;
            }
            CPlaceInner::Addr(ptr, None) => ptr,
            CPlaceInner::NoPlace => {
                if dst_layout.abi != Abi::Uninhabited {
                    assert_eq!(dst_layout.size.bytes(), 0, "{:?}", dst_layout);
                }
                return;
            }
            CPlaceInner::Addr(_, Some(_)) => bug!("Can't write value to unsized place {:?}", self),
        };

        match self.layout().abi {
            // FIXME make Abi::Vector work too
            Abi::Scalar(_) => {
                let val = from.load_scalar(fx);
                to_ptr.store(fx, val, MemFlags::new());
                return;
            }
            Abi::ScalarPair(ref a_scalar, ref b_scalar) => {
                let (value, extra) = from.load_scalar_pair(fx);
                let b_offset = scalar_pair_calculate_b_offset(fx.tcx, a_scalar, b_scalar);
                to_ptr.store(fx, value, MemFlags::new());
                to_ptr.offset(fx, b_offset).store(fx, extra, MemFlags::new());
                return;
            }
            _ => {}
        }

        match from.0 {
            CValueInner::ByVal(val) => {
                to_ptr.store(fx, val, MemFlags::new());
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
                    fx.module.target_config(),
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

    pub fn place_field(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
        field: mir::Field,
    ) -> CPlace<'tcx> {
        let layout = self.layout();
        let (base, extra) = self.to_ptr_maybe_unsized(fx);

        let (field_ptr, field_layout) = codegen_field(fx, base, extra, layout, field);
        if field_layout.is_unsized() {
            CPlace::for_ptr_with_extra(field_ptr, extra.unwrap(), field_layout)
        } else {
            CPlace::for_ptr(field_ptr, field_layout)
        }
    }

    pub fn place_index(
        self,
        fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
        index: Value,
    ) -> CPlace<'tcx> {
        let (elem_layout, ptr) = match self.layout().ty.kind {
            ty::Array(elem_ty, _) => (fx.layout_of(elem_ty), self.to_ptr(fx)),
            ty::Slice(elem_ty) => (fx.layout_of(elem_ty), self.to_ptr_maybe_unsized(fx).0),
            _ => bug!("place_index({:?})", self.layout().ty),
        };

        let offset = fx
            .bcx
            .ins()
            .imul_imm(index, elem_layout.size.bytes() as i64);

        CPlace::for_ptr(ptr.offset_value(fx, offset), elem_layout)
    }

    pub fn place_deref(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>) -> CPlace<'tcx> {
        let inner_layout = fx.layout_of(self.layout().ty.builtin_deref(true).unwrap().ty);
        if has_ptr_meta(fx.tcx, inner_layout.ty) {
            let (addr, extra) = self.to_cvalue(fx).load_scalar_pair(fx);
            CPlace::for_ptr_with_extra(Pointer::new(addr), extra, inner_layout)
        } else {
            CPlace::for_ptr(Pointer::new(self.to_cvalue(fx).load_scalar(fx)), inner_layout)
        }
    }

    pub fn write_place_ref(self, fx: &mut FunctionCx<'_, 'tcx, impl Backend>, dest: CPlace<'tcx>) {
        if has_ptr_meta(fx.tcx, self.layout().ty) {
            let (ptr, extra) = self.to_ptr_maybe_unsized(fx);
            let ptr = CValue::by_val_pair(
                ptr.get_addr(fx),
                extra.expect("unsized type without metadata"),
                dest.layout(),
            );
            dest.write_cvalue(fx, ptr);
        } else {
            let ptr = CValue::by_val(self.to_ptr(fx).get_addr(fx), dest.layout());
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
