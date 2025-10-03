//! Definition of [`CValue`] and [`CPlace`]

use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_frontend::Variable;
use rustc_middle::ty::FnSig;
use rustc_middle::ty::layout::HasTypingEnv;

use crate::prelude::*;

fn codegen_field<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    base: Pointer,
    extra: Option<Value>,
    layout: TyAndLayout<'tcx>,
    field: FieldIdx,
) -> (Pointer, TyAndLayout<'tcx>) {
    let field_offset = layout.fields.offset(field.index());
    let field_layout = layout.field(&*fx, field.index());

    let simple = |fx: &mut FunctionCx<'_, '_, '_>| {
        (base.offset_i64(fx, i64::try_from(field_offset.bytes()).unwrap()), field_layout)
    };

    if field_layout.is_sized() {
        return simple(fx);
    }
    match field_layout.ty.kind() {
        ty::Slice(..) | ty::Str => simple(fx),
        _ => {
            let unaligned_offset = field_offset.bytes();

            // Get the alignment of the field
            let (_, mut unsized_align) = crate::unsize::size_and_align_of(fx, field_layout, extra);

            // For packed types, we need to cap alignment.
            if let ty::Adt(def, _) = layout.ty.kind() {
                if let Some(packed) = def.repr().pack {
                    let packed = fx.bcx.ins().iconst(fx.pointer_type, packed.bytes() as i64);
                    let cmp = fx.bcx.ins().icmp(IntCC::UnsignedLessThan, unsized_align, packed);
                    unsized_align = fx.bcx.ins().select(cmp, unsized_align, packed);
                }
            }

            // Bump the unaligned offset up to the appropriate alignment
            let one = fx.bcx.ins().iconst(fx.pointer_type, 1);
            let align_sub_1 = fx.bcx.ins().isub(unsized_align, one);
            let and_lhs = fx.bcx.ins().iadd_imm(align_sub_1, unaligned_offset as i64);
            let zero = fx.bcx.ins().iconst(fx.pointer_type, 0);
            let and_rhs = fx.bcx.ins().isub(zero, unsized_align);
            let offset = fx.bcx.ins().band(and_lhs, and_rhs);

            (base.offset_value(fx, offset), field_layout)
        }
    }
}

fn scalar_pair_calculate_b_offset(tcx: TyCtxt<'_>, a_scalar: Scalar, b_scalar: Scalar) -> Offset32 {
    let b_offset = a_scalar.size(&tcx).align_to(b_scalar.align(&tcx).abi);
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

    /// Create an instance of a ZST
    ///
    /// The is represented by a dangling pointer of suitable alignment.
    pub(crate) fn zst(layout: TyAndLayout<'tcx>) -> CValue<'tcx> {
        assert!(layout.is_zst());
        CValue::by_ref(crate::Pointer::dangling(layout.align.abi), layout)
    }

    pub(crate) fn layout(&self) -> TyAndLayout<'tcx> {
        self.1
    }

    // FIXME remove
    pub(crate) fn force_stack(self, fx: &mut FunctionCx<'_, '_, 'tcx>) -> (Pointer, Option<Value>) {
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

    /// Load a value with layout.backend_repr of scalar
    #[track_caller]
    pub(crate) fn load_scalar(self, fx: &mut FunctionCx<'_, '_, 'tcx>) -> Value {
        let layout = self.1;
        match self.0 {
            CValueInner::ByRef(ptr, None) => {
                let clif_ty = match layout.backend_repr {
                    BackendRepr::Scalar(scalar) => scalar_to_clif_type(fx.tcx, scalar),
                    BackendRepr::SimdVector { element, count } => {
                        scalar_to_clif_type(fx.tcx, element)
                            .by(u32::try_from(count).unwrap())
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

    /// Load a value pair with layout.backend_repr of scalar pair
    #[track_caller]
    pub(crate) fn load_scalar_pair(self, fx: &mut FunctionCx<'_, '_, 'tcx>) -> (Value, Value) {
        let layout = self.1;
        match self.0 {
            CValueInner::ByRef(ptr, None) => {
                let (a_scalar, b_scalar) = match layout.backend_repr {
                    BackendRepr::ScalarPair(a, b) => (a, b),
                    _ => unreachable!("load_scalar_pair({:?})", self),
                };
                let b_offset = scalar_pair_calculate_b_offset(fx.tcx, a_scalar, b_scalar);
                let clif_ty1 = scalar_to_clif_type(fx.tcx, a_scalar);
                let clif_ty2 = scalar_to_clif_type(fx.tcx, b_scalar);
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
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        field: FieldIdx,
    ) -> CValue<'tcx> {
        let layout = self.1;
        match self.0 {
            CValueInner::ByVal(_) => unreachable!(),
            CValueInner::ByValPair(val1, val2) => match layout.backend_repr {
                BackendRepr::ScalarPair(_, _) => {
                    let val = match field.as_u32() {
                        0 => val1,
                        1 => val2,
                        _ => bug!("field should be 0 or 1"),
                    };
                    let field_layout = layout.field(&*fx, usize::from(field));
                    CValue::by_val(val, field_layout)
                }
                _ => unreachable!("value_field for ByValPair with abi {:?}", layout.backend_repr),
            },
            CValueInner::ByRef(ptr, None) => {
                let (field_ptr, field_layout) = codegen_field(fx, ptr, None, layout, field);
                CValue::by_ref(field_ptr, field_layout)
            }
            CValueInner::ByRef(_, Some(_)) => todo!(),
        }
    }

    /// Like [`CValue::value_field`] except handling ADTs containing a single array field in a way
    /// such that you can access individual lanes.
    pub(crate) fn value_lane(
        self,
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        lane_idx: u64,
    ) -> CValue<'tcx> {
        let layout = self.1;
        assert!(layout.ty.is_simd());
        let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
        let lane_layout = fx.layout_of(lane_ty);
        assert!(lane_idx < lane_count);

        match self.0 {
            CValueInner::ByVal(_) | CValueInner::ByValPair(_, _) => unreachable!(),
            CValueInner::ByRef(ptr, None) => {
                let field_offset = lane_layout.size * lane_idx;
                let field_ptr = ptr.offset_i64(fx, i64::try_from(field_offset.bytes()).unwrap());
                CValue::by_ref(field_ptr, lane_layout)
            }
            CValueInner::ByRef(_, Some(_)) => unreachable!(),
        }
    }

    /// Like [`CValue::value_field`] except using the passed type as lane type instead of the one
    /// specified by the vector type.
    pub(crate) fn value_typed_lane(
        self,
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        lane_ty: Ty<'tcx>,
        lane_idx: u64,
    ) -> CValue<'tcx> {
        let layout = self.1;
        assert!(layout.ty.is_simd());
        let (orig_lane_count, orig_lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
        let lane_layout = fx.layout_of(lane_ty);
        assert!(
            (lane_idx + 1) * lane_layout.size <= orig_lane_count * fx.layout_of(orig_lane_ty).size
        );

        match self.0 {
            CValueInner::ByVal(_) | CValueInner::ByValPair(_, _) => unreachable!(),
            CValueInner::ByRef(ptr, None) => {
                let field_offset = lane_layout.size * lane_idx;
                let field_ptr = ptr.offset_i64(fx, i64::try_from(field_offset.bytes()).unwrap());
                CValue::by_ref(field_ptr, lane_layout)
            }
            CValueInner::ByRef(_, Some(_)) => unreachable!(),
        }
    }

    /// Like [`CValue::value_lane`] except allowing a dynamically calculated lane index.
    pub(crate) fn value_lane_dyn(
        self,
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        lane_idx: Value,
    ) -> CValue<'tcx> {
        let layout = self.1;
        assert!(layout.ty.is_simd());
        let (_lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
        let lane_layout = fx.layout_of(lane_ty);
        match self.0 {
            CValueInner::ByVal(_) | CValueInner::ByValPair(_, _) => unreachable!(),
            CValueInner::ByRef(ptr, None) => {
                let lane_idx = clif_intcast(fx, lane_idx, fx.pointer_type, false);
                let field_offset = fx.bcx.ins().imul_imm(lane_idx, lane_layout.size.bytes() as i64);
                let field_ptr = ptr.offset_value(fx, field_offset);
                CValue::by_ref(field_ptr, lane_layout)
            }
            CValueInner::ByRef(_, Some(_)) => unreachable!(),
        }
    }

    /// If `ty` is signed, `const_val` must already be sign extended.
    pub(crate) fn const_val(
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        layout: TyAndLayout<'tcx>,
        const_val: ty::ScalarInt,
    ) -> CValue<'tcx> {
        assert_eq!(const_val.size(), layout.size, "{:#?}: {:?}", const_val, layout);
        use cranelift_codegen::ir::immediates::{Ieee16, Ieee32, Ieee64, Ieee128};

        let clif_ty = fx.clif_type(layout.ty).unwrap();

        let val = match layout.ty.kind() {
            ty::Uint(UintTy::U128) | ty::Int(IntTy::I128) => {
                let const_val = const_val.to_bits(layout.size);
                let lsb = fx.bcx.ins().iconst(types::I64, const_val as u64 as i64);
                let msb = fx.bcx.ins().iconst(types::I64, (const_val >> 64) as u64 as i64);
                fx.bcx.ins().iconcat(lsb, msb)
            }
            ty::Bool
            | ty::Char
            | ty::Uint(_)
            | ty::Int(_)
            | ty::Ref(..)
            | ty::RawPtr(..)
            | ty::FnPtr(..) => {
                let raw_val = const_val.size().truncate(const_val.to_bits(layout.size));
                fx.bcx.ins().iconst(clif_ty, raw_val as i64)
            }
            ty::Float(FloatTy::F16) => {
                fx.bcx.ins().f16const(Ieee16::with_bits(u16::try_from(const_val).unwrap()))
            }
            ty::Float(FloatTy::F32) => {
                fx.bcx.ins().f32const(Ieee32::with_bits(u32::try_from(const_val).unwrap()))
            }
            ty::Float(FloatTy::F64) => {
                fx.bcx.ins().f64const(Ieee64::with_bits(u64::try_from(const_val).unwrap()))
            }
            ty::Float(FloatTy::F128) => {
                let value = fx
                    .bcx
                    .func
                    .dfg
                    .constants
                    .insert(Ieee128::with_bits(u128::try_from(const_val).unwrap()).into());
                fx.bcx.ins().f128const(value)
            }
            _ => panic!(
                "CValue::const_val for non bool/char/float/integer/pointer type {:?} is not allowed",
                layout.ty
            ),
        };

        CValue::by_val(val, layout)
    }

    pub(crate) fn cast_pointer_to(self, layout: TyAndLayout<'tcx>) -> Self {
        assert!(matches!(self.layout().ty.kind(), ty::Ref(..) | ty::RawPtr(..) | ty::FnPtr(..)));
        assert!(matches!(layout.ty.kind(), ty::Ref(..) | ty::RawPtr(..) | ty::FnPtr(..)));
        assert_eq!(self.layout().backend_repr, layout.backend_repr);
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
enum CPlaceInner {
    Var(Local, Variable),
    VarPair(Local, Variable, Variable),
    Addr(Pointer, Option<Value>),
}

impl<'tcx> CPlace<'tcx> {
    pub(crate) fn layout(&self) -> TyAndLayout<'tcx> {
        self.layout
    }

    pub(crate) fn new_stack_slot(
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        layout: TyAndLayout<'tcx>,
    ) -> CPlace<'tcx> {
        assert!(layout.is_sized());
        if layout.size.bytes() == 0 {
            return CPlace {
                inner: CPlaceInner::Addr(Pointer::dangling(layout.align.abi), None),
                layout,
            };
        }

        if layout.size.bytes() >= u64::from(u32::MAX - 16) {
            fx.tcx
                .dcx()
                .fatal(format!("values of type {} are too big to store on the stack", layout.ty));
        }

        let stack_slot = fx.create_stack_slot(
            u32::try_from(layout.size.bytes()).unwrap(),
            u32::try_from(layout.align.bytes()).unwrap(),
        );
        CPlace { inner: CPlaceInner::Addr(stack_slot, None), layout }
    }

    pub(crate) fn new_var(
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        local: Local,
        layout: TyAndLayout<'tcx>,
    ) -> CPlace<'tcx> {
        let var = Variable::from_u32(fx.next_ssa_var);
        fx.next_ssa_var += 1;
        fx.bcx.declare_var(var, fx.clif_type(layout.ty).unwrap());
        CPlace { inner: CPlaceInner::Var(local, var), layout }
    }

    pub(crate) fn new_var_pair(
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        local: Local,
        layout: TyAndLayout<'tcx>,
    ) -> CPlace<'tcx> {
        let var1 = Variable::from_u32(fx.next_ssa_var);
        fx.next_ssa_var += 1;
        let var2 = Variable::from_u32(fx.next_ssa_var);
        fx.next_ssa_var += 1;

        let (ty1, ty2) = fx.clif_pair_type(layout.ty).unwrap();
        fx.bcx.declare_var(var1, ty1);
        fx.bcx.declare_var(var2, ty2);
        CPlace { inner: CPlaceInner::VarPair(local, var1, var2), layout }
    }

    pub(crate) fn for_ptr(ptr: Pointer, layout: TyAndLayout<'tcx>) -> CPlace<'tcx> {
        CPlace { inner: CPlaceInner::Addr(ptr, None), layout }
    }

    pub(crate) fn for_ptr_with_extra(
        ptr: Pointer,
        extra: Value,
        layout: TyAndLayout<'tcx>,
    ) -> CPlace<'tcx> {
        CPlace { inner: CPlaceInner::Addr(ptr, Some(extra)), layout }
    }

    pub(crate) fn to_cvalue(self, fx: &mut FunctionCx<'_, '_, 'tcx>) -> CValue<'tcx> {
        let layout = self.layout();
        match self.inner {
            CPlaceInner::Var(_local, var) => {
                let val = fx.bcx.use_var(var);
                //fx.bcx.set_val_label(val, cranelift_codegen::ir::ValueLabel::new(var.index()));
                CValue::by_val(val, layout)
            }
            CPlaceInner::VarPair(_local, var1, var2) => {
                let val1 = fx.bcx.use_var(var1);
                //fx.bcx.set_val_label(val1, cranelift_codegen::ir::ValueLabel::new(var1.index()));
                let val2 = fx.bcx.use_var(var2);
                //fx.bcx.set_val_label(val2, cranelift_codegen::ir::ValueLabel::new(var2.index()));
                CValue::by_val_pair(val1, val2, layout)
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

    pub(crate) fn debug_comment(self) -> (&'static str, String) {
        match self.inner {
            CPlaceInner::Var(_local, var) => ("ssa", format!("var={}", var.index())),
            CPlaceInner::VarPair(_local, var1, var2) => {
                ("ssa", format!("var=({}, {})", var1.index(), var2.index()))
            }
            CPlaceInner::Addr(ptr, meta) => {
                let meta =
                    if let Some(meta) = meta { format!(",meta={}", meta) } else { String::new() };
                match ptr.debug_base_and_offset() {
                    (crate::pointer::PointerBase::Addr(addr), offset) => {
                        ("reuse", format!("storage={}{}{}", addr, offset, meta))
                    }
                    (crate::pointer::PointerBase::Stack(stack_slot), offset) => {
                        ("stack", format!("storage={}{}{}", stack_slot, offset, meta))
                    }
                    (crate::pointer::PointerBase::Dangling(align), offset) => {
                        ("zst", format!("align={},offset={}", align.bytes(), offset))
                    }
                }
            }
        }
    }

    #[track_caller]
    pub(crate) fn to_ptr(self) -> Pointer {
        match self.inner {
            CPlaceInner::Addr(ptr, None) => ptr,
            CPlaceInner::Addr(_, Some(_)) => bug!("Expected sized cplace, found {:?}", self),
            CPlaceInner::Var(_, _) | CPlaceInner::VarPair(_, _, _) => {
                bug!("Expected CPlace::Addr, found {:?}", self)
            }
        }
    }

    #[track_caller]
    pub(crate) fn to_ptr_unsized(self) -> (Pointer, Value) {
        match self.inner {
            CPlaceInner::Addr(ptr, Some(extra)) => (ptr, extra),
            CPlaceInner::Addr(_, None) | CPlaceInner::Var(_, _) | CPlaceInner::VarPair(_, _, _) => {
                bug!("Expected unsized cplace, found {:?}", self)
            }
        }
    }

    pub(crate) fn try_to_ptr(self) -> Option<Pointer> {
        match self.inner {
            CPlaceInner::Var(_, _) | CPlaceInner::VarPair(_, _, _) => None,
            CPlaceInner::Addr(ptr, None) => Some(ptr),
            CPlaceInner::Addr(_, Some(_)) => bug!("Expected sized cplace, found {:?}", self),
        }
    }

    pub(crate) fn write_cvalue(self, fx: &mut FunctionCx<'_, '_, 'tcx>, from: CValue<'tcx>) {
        assert_assignable(fx, from.layout().ty, self.layout().ty, 16);

        self.write_cvalue_maybe_transmute(fx, from, "write_cvalue");
    }

    pub(crate) fn write_cvalue_transmute(
        self,
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        from: CValue<'tcx>,
    ) {
        self.write_cvalue_maybe_transmute(fx, from, "write_cvalue_transmute");
    }

    fn write_cvalue_maybe_transmute(
        self,
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        from: CValue<'tcx>,
        method: &'static str,
    ) {
        fn transmute_scalar<'tcx>(
            fx: &mut FunctionCx<'_, '_, 'tcx>,
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
            let data = if src_ty == dst_ty { data } else { codegen_bitcast(fx, dst_ty, data) };
            //fx.bcx.set_val_label(data, cranelift_codegen::ir::ValueLabel::new(var.index()));
            fx.bcx.def_var(var, data);
        }

        assert_eq!(self.layout().size, from.layout().size);

        if fx.clif_comments.enabled() {
            let inst = fx.bcx.func.layout.last_inst(fx.bcx.current_block().unwrap()).unwrap();
            fx.add_post_comment(
                inst,
                format!(
                    "{}: {:?}: {:?} <- {:?}: {:?}",
                    method,
                    self.inner,
                    self.layout().ty,
                    from.0,
                    from.layout().ty
                ),
            );
        }

        let dst_layout = self.layout();
        match self.inner {
            CPlaceInner::Var(_local, var) => {
                let data = match from.1.backend_repr {
                    BackendRepr::Scalar(_) => CValue(from.0, dst_layout).load_scalar(fx),
                    _ => {
                        let (ptr, meta) = from.force_stack(fx);
                        assert!(meta.is_none());
                        CValue(CValueInner::ByRef(ptr, None), dst_layout).load_scalar(fx)
                    }
                };
                let dst_ty = fx.clif_type(self.layout().ty).unwrap();
                transmute_scalar(fx, var, data, dst_ty);
            }
            CPlaceInner::VarPair(_local, var1, var2) => {
                let (data1, data2) = match from.1.backend_repr {
                    BackendRepr::ScalarPair(_, _) => {
                        CValue(from.0, dst_layout).load_scalar_pair(fx)
                    }
                    _ => {
                        let (ptr, meta) = from.force_stack(fx);
                        assert!(meta.is_none());
                        CValue(CValueInner::ByRef(ptr, None), dst_layout).load_scalar_pair(fx)
                    }
                };
                let (dst_ty1, dst_ty2) = fx.clif_pair_type(self.layout().ty).unwrap();
                transmute_scalar(fx, var1, data1, dst_ty1);
                transmute_scalar(fx, var2, data2, dst_ty2);
            }
            CPlaceInner::Addr(_, Some(_)) => bug!("Can't write value to unsized place {:?}", self),
            CPlaceInner::Addr(to_ptr, None) => {
                if dst_layout.size == Size::ZERO {
                    return;
                }

                let mut flags = MemFlags::new();
                flags.set_notrap();

                match from.0 {
                    CValueInner::ByVal(val) => {
                        to_ptr.store(fx, val, flags);
                    }
                    CValueInner::ByValPair(val1, val2) => match from.layout().backend_repr {
                        BackendRepr::ScalarPair(a_scalar, b_scalar) => {
                            let b_offset =
                                scalar_pair_calculate_b_offset(fx.tcx, a_scalar, b_scalar);
                            to_ptr.store(fx, val1, flags);
                            to_ptr.offset(fx, b_offset).store(fx, val2, flags);
                        }
                        _ => {
                            bug!(
                                "Non ScalarPair repr {:?} for ByValPair CValue",
                                dst_layout.backend_repr
                            )
                        }
                    },
                    CValueInner::ByRef(from_ptr, None) => {
                        match from.layout().backend_repr {
                            BackendRepr::Scalar(_) => {
                                let val = from.load_scalar(fx);
                                to_ptr.store(fx, val, flags);
                                return;
                            }
                            BackendRepr::ScalarPair(a_scalar, b_scalar) => {
                                let b_offset =
                                    scalar_pair_calculate_b_offset(fx.tcx, a_scalar, b_scalar);
                                let (val1, val2) = from.load_scalar_pair(fx);
                                to_ptr.store(fx, val1, flags);
                                to_ptr.offset(fx, b_offset).store(fx, val2, flags);
                                return;
                            }
                            _ => {}
                        }

                        let from_addr = from_ptr.get_addr(fx);
                        let to_addr = to_ptr.get_addr(fx);
                        let src_layout = from.1;
                        let size = dst_layout.size.bytes();
                        // `emit_small_memory_copy` uses `u8` for alignments, just use the maximum
                        // alignment that fits in a `u8` if the actual alignment is larger.
                        let src_align = src_layout.align.bytes().try_into().unwrap_or(128);
                        let dst_align = dst_layout.align.bytes().try_into().unwrap_or(128);
                        fx.bcx.emit_small_memory_copy(
                            fx.target_config,
                            to_addr,
                            from_addr,
                            size,
                            dst_align,
                            src_align,
                            true,
                            flags,
                        );
                    }
                    CValueInner::ByRef(_, Some(_)) => todo!(),
                }
            }
        }
    }

    /// Used for `ProjectionElem::UnwrapUnsafeBinder`, `ty` has to be monomorphized before
    /// passed on.
    pub(crate) fn place_transmute_type(
        self,
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        ty: Ty<'tcx>,
    ) -> CPlace<'tcx> {
        CPlace { inner: self.inner, layout: fx.layout_of(ty) }
    }

    pub(crate) fn place_field(
        self,
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        field: FieldIdx,
    ) -> CPlace<'tcx> {
        let layout = self.layout();

        match self.inner {
            CPlaceInner::VarPair(local, var1, var2) => {
                let layout = layout.field(&*fx, field.index());

                match field.as_u32() {
                    0 => return CPlace { inner: CPlaceInner::Var(local, var1), layout },
                    1 => return CPlace { inner: CPlaceInner::Var(local, var2), layout },
                    _ => unreachable!("field should be 0 or 1"),
                }
            }
            _ => {}
        }

        let (base, extra) = match self.inner {
            CPlaceInner::Addr(ptr, extra) => (ptr, extra),
            CPlaceInner::Var(_, _) | CPlaceInner::VarPair(_, _, _) => {
                bug!("Expected CPlace::Addr, found {:?}", self)
            }
        };

        let (field_ptr, field_layout) = codegen_field(fx, base, extra, layout, field);
        if fx.tcx.type_has_metadata(field_layout.ty, ty::TypingEnv::fully_monomorphized()) {
            CPlace::for_ptr_with_extra(field_ptr, extra.unwrap(), field_layout)
        } else {
            CPlace::for_ptr(field_ptr, field_layout)
        }
    }

    /// Like [`CPlace::place_field`] except handling ADTs containing a single array field in a way
    /// such that you can access individual lanes.
    pub(crate) fn place_lane(
        self,
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        lane_idx: u64,
    ) -> CPlace<'tcx> {
        let layout = self.layout();
        assert!(layout.ty.is_simd());
        let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
        let lane_layout = fx.layout_of(lane_ty);
        assert!(lane_idx < lane_count);

        match self.inner {
            CPlaceInner::Var(_, _) => unreachable!(),
            CPlaceInner::VarPair(_, _, _) => unreachable!(),
            CPlaceInner::Addr(ptr, None) => {
                let field_offset = lane_layout.size * lane_idx;
                let field_ptr = ptr.offset_i64(fx, i64::try_from(field_offset.bytes()).unwrap());
                CPlace::for_ptr(field_ptr, lane_layout)
            }
            CPlaceInner::Addr(_, Some(_)) => unreachable!(),
        }
    }

    /// Like [`CPlace::place_field`] except using the passed type as lane type instead of the one
    /// specified by the vector type.
    pub(crate) fn place_typed_lane(
        self,
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        lane_ty: Ty<'tcx>,
        lane_idx: u64,
    ) -> CPlace<'tcx> {
        let layout = self.layout();
        assert!(layout.ty.is_simd());
        let (orig_lane_count, orig_lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
        let lane_layout = fx.layout_of(lane_ty);
        assert!(
            (lane_idx + 1) * lane_layout.size <= orig_lane_count * fx.layout_of(orig_lane_ty).size
        );

        match self.inner {
            CPlaceInner::Var(_, _) => unreachable!(),
            CPlaceInner::VarPair(_, _, _) => unreachable!(),
            CPlaceInner::Addr(ptr, None) => {
                let field_offset = lane_layout.size * lane_idx;
                let field_ptr = ptr.offset_i64(fx, i64::try_from(field_offset.bytes()).unwrap());
                CPlace::for_ptr(field_ptr, lane_layout)
            }
            CPlaceInner::Addr(_, Some(_)) => unreachable!(),
        }
    }

    /// Write a value to an individual lane in a SIMD vector.
    pub(crate) fn write_lane_dyn(
        self,
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        lane_idx: Value,
        value: CValue<'tcx>,
    ) {
        let layout = self.layout();
        assert!(layout.ty.is_simd());
        let (_lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
        let lane_layout = fx.layout_of(lane_ty);
        assert_eq!(lane_layout, value.layout());

        match self.inner {
            CPlaceInner::Var(_, _) => unreachable!(),
            CPlaceInner::VarPair(_, _, _) => unreachable!(),
            CPlaceInner::Addr(ptr, None) => {
                let lane_idx = clif_intcast(fx, lane_idx, fx.pointer_type, false);
                let field_offset = fx
                    .bcx
                    .ins()
                    .imul_imm(lane_idx, i64::try_from(lane_layout.size.bytes()).unwrap());
                let field_ptr = ptr.offset_value(fx, field_offset);
                CPlace::for_ptr(field_ptr, lane_layout).write_cvalue(fx, value);
            }
            CPlaceInner::Addr(_, Some(_)) => unreachable!(),
        }
    }

    pub(crate) fn place_index(
        self,
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        index: Value,
    ) -> CPlace<'tcx> {
        let (elem_layout, ptr) = match self.layout().ty.kind() {
            ty::Array(elem_ty, _) => {
                let elem_layout = fx.layout_of(*elem_ty);
                match self.inner {
                    CPlaceInner::Addr(addr, None) => (elem_layout, addr),
                    CPlaceInner::Var(_, _)
                    | CPlaceInner::Addr(_, Some(_))
                    | CPlaceInner::VarPair(_, _, _) => bug!("Can't index into {self:?}"),
                }
            }
            ty::Slice(elem_ty) => (fx.layout_of(*elem_ty), self.to_ptr_unsized().0),
            _ => bug!("place_index({:?})", self.layout().ty),
        };

        let offset = fx.bcx.ins().imul_imm(index, elem_layout.size.bytes() as i64);

        CPlace::for_ptr(ptr.offset_value(fx, offset), elem_layout)
    }

    pub(crate) fn place_deref(self, fx: &mut FunctionCx<'_, '_, 'tcx>) -> CPlace<'tcx> {
        let inner_layout = fx.layout_of(self.layout().ty.builtin_deref(true).unwrap());
        if fx.tcx.type_has_metadata(inner_layout.ty, ty::TypingEnv::fully_monomorphized()) {
            let (addr, extra) = self.to_cvalue(fx).load_scalar_pair(fx);
            CPlace::for_ptr_with_extra(Pointer::new(addr), extra, inner_layout)
        } else {
            CPlace::for_ptr(Pointer::new(self.to_cvalue(fx).load_scalar(fx)), inner_layout)
        }
    }

    pub(crate) fn place_ref(
        self,
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        layout: TyAndLayout<'tcx>,
    ) -> CValue<'tcx> {
        if fx.tcx.type_has_metadata(self.layout().ty, ty::TypingEnv::fully_monomorphized()) {
            let (ptr, extra) = self.to_ptr_unsized();
            CValue::by_val_pair(ptr.get_addr(fx), extra, layout)
        } else {
            CValue::by_val(self.to_ptr().get_addr(fx), layout)
        }
    }

    pub(crate) fn downcast_variant(
        self,
        fx: &FunctionCx<'_, '_, 'tcx>,
        variant: VariantIdx,
    ) -> Self {
        assert!(self.layout().is_sized());
        let layout = self.layout().for_variant(fx, variant);
        CPlace { inner: self.inner, layout }
    }
}

#[track_caller]
pub(crate) fn assert_assignable<'tcx>(
    fx: &FunctionCx<'_, '_, 'tcx>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    limit: usize,
) {
    if limit == 0 {
        // assert_assignable exists solely to catch bugs in cg_clif. it isn't necessary for
        // soundness. don't attempt to check deep types to avoid exponential behavior in certain
        // cases.
        return;
    }
    match (from_ty.kind(), to_ty.kind()) {
        (ty::Ref(_, a, _), ty::Ref(_, b, _)) | (ty::RawPtr(a, _), ty::RawPtr(b, _)) => {
            assert_assignable(fx, *a, *b, limit - 1);
        }
        (ty::Ref(_, a, _), ty::RawPtr(b, _)) | (ty::RawPtr(a, _), ty::Ref(_, b, _)) => {
            assert_assignable(fx, *a, *b, limit - 1);
        }
        (ty::FnPtr(..), ty::FnPtr(..)) => {
            let from_sig = fx
                .tcx
                .normalize_erasing_late_bound_regions(fx.typing_env(), from_ty.fn_sig(fx.tcx));
            let FnSig {
                inputs_and_output: types_from,
                c_variadic: c_variadic_from,
                safety: unsafety_from,
                abi: abi_from,
            } = from_sig;
            let to_sig =
                fx.tcx.normalize_erasing_late_bound_regions(fx.typing_env(), to_ty.fn_sig(fx.tcx));
            let FnSig {
                inputs_and_output: types_to,
                c_variadic: c_variadic_to,
                safety: unsafety_to,
                abi: abi_to,
            } = to_sig;
            let mut types_from = types_from.iter();
            let mut types_to = types_to.iter();
            loop {
                match (types_from.next(), types_to.next()) {
                    (Some(a), Some(b)) => assert_assignable(fx, a, b, limit - 1),
                    (None, None) => break,
                    (Some(_), None) | (None, Some(_)) => panic!("{:#?}/{:#?}", from_ty, to_ty),
                }
            }
            assert_eq!(
                c_variadic_from, c_variadic_to,
                "Can't write fn ptr with incompatible sig {:?} to place with sig {:?}\n\n{:#?}",
                from_sig, to_sig, fx,
            );
            assert_eq!(
                unsafety_from, unsafety_to,
                "Can't write fn ptr with incompatible sig {:?} to place with sig {:?}\n\n{:#?}",
                from_sig, to_sig, fx,
            );
            assert_eq!(
                abi_from, abi_to,
                "Can't write fn ptr with incompatible sig {:?} to place with sig {:?}\n\n{:#?}",
                from_sig, to_sig, fx,
            );
            // fn(&T) -> for<'l> fn(&'l T) is allowed
        }
        (&ty::Dynamic(from_traits, _), &ty::Dynamic(to_traits, _)) => {
            for (from, to) in from_traits.iter().zip(to_traits) {
                let from = fx.tcx.normalize_erasing_late_bound_regions(fx.typing_env(), from);
                let to = fx.tcx.normalize_erasing_late_bound_regions(fx.typing_env(), to);
                assert_eq!(
                    from, to,
                    "Can't write trait object of incompatible traits {:?} to place with traits {:?}\n\n{:#?}",
                    from_traits, to_traits, fx,
                );
            }
            // dyn for<'r> Trait<'r> -> dyn Trait<'_> is allowed
        }
        (&ty::Tuple(types_a), &ty::Tuple(types_b)) => {
            let mut types_a = types_a.iter();
            let mut types_b = types_b.iter();
            loop {
                match (types_a.next(), types_b.next()) {
                    (Some(a), Some(b)) => assert_assignable(fx, a, b, limit - 1),
                    (None, None) => return,
                    (Some(_), None) | (None, Some(_)) => panic!("{:#?}/{:#?}", from_ty, to_ty),
                }
            }
        }
        (&ty::Adt(adt_def_a, args_a), &ty::Adt(adt_def_b, args_b))
            if adt_def_a.did() == adt_def_b.did() =>
        {
            let mut types_a = args_a.types();
            let mut types_b = args_b.types();
            loop {
                match (types_a.next(), types_b.next()) {
                    (Some(a), Some(b)) => assert_assignable(fx, a, b, limit - 1),
                    (None, None) => return,
                    (Some(_), None) | (None, Some(_)) => panic!("{:#?}/{:#?}", from_ty, to_ty),
                }
            }
        }
        (ty::Array(a, _), ty::Array(b, _)) => assert_assignable(fx, *a, *b, limit - 1),
        (&ty::Closure(def_id_a, args_a), &ty::Closure(def_id_b, args_b))
            if def_id_a == def_id_b =>
        {
            let mut types_a = args_a.types();
            let mut types_b = args_b.types();
            loop {
                match (types_a.next(), types_b.next()) {
                    (Some(a), Some(b)) => assert_assignable(fx, a, b, limit - 1),
                    (None, None) => return,
                    (Some(_), None) | (None, Some(_)) => panic!("{:#?}/{:#?}", from_ty, to_ty),
                }
            }
        }
        (&ty::Coroutine(def_id_a, args_a), &ty::Coroutine(def_id_b, args_b))
            if def_id_a == def_id_b =>
        {
            let mut types_a = args_a.types();
            let mut types_b = args_b.types();
            loop {
                match (types_a.next(), types_b.next()) {
                    (Some(a), Some(b)) => assert_assignable(fx, a, b, limit - 1),
                    (None, None) => return,
                    (Some(_), None) | (None, Some(_)) => panic!("{:#?}/{:#?}", from_ty, to_ty),
                }
            }
        }
        (&ty::CoroutineWitness(def_id_a, args_a), &ty::CoroutineWitness(def_id_b, args_b))
            if def_id_a == def_id_b =>
        {
            let mut types_a = args_a.types();
            let mut types_b = args_b.types();
            loop {
                match (types_a.next(), types_b.next()) {
                    (Some(a), Some(b)) => assert_assignable(fx, a, b, limit - 1),
                    (None, None) => return,
                    (Some(_), None) | (None, Some(_)) => panic!("{:#?}/{:#?}", from_ty, to_ty),
                }
            }
        }
        _ => {
            assert_eq!(
                from_ty,
                to_ty,
                "Can't write value with incompatible type {:?} to place with type {:?}\n\n{:#?}",
                from_ty.kind(),
                to_ty.kind(),
                fx,
            );
        }
    }
}
