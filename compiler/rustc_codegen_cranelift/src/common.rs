use cranelift_codegen::isa::TargetFrontendConfig;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use rustc_abi::{Float, Integer, Primitive};
use rustc_index::IndexVec;
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::layout::{
    self, FnAbiError, FnAbiOfHelpers, FnAbiRequest, LayoutError, LayoutOfHelpers,
};
use rustc_span::source_map::Spanned;
use rustc_target::callconv::FnAbi;
use rustc_target::spec::{HasTargetSpec, Target};

use crate::constant::ConstantCx;
use crate::debuginfo::FunctionDebugContext;
use crate::prelude::*;

pub(crate) fn pointer_ty(tcx: TyCtxt<'_>) -> types::Type {
    match tcx.data_layout.pointer_size().bits() {
        16 => types::I16,
        32 => types::I32,
        64 => types::I64,
        bits => bug!("ptr_sized_integer: unknown pointer bit size {}", bits),
    }
}

pub(crate) fn scalar_to_clif_type(tcx: TyCtxt<'_>, scalar: Scalar) -> Type {
    match scalar.primitive() {
        Primitive::Int(int, _sign) => match int {
            Integer::I8 => types::I8,
            Integer::I16 => types::I16,
            Integer::I32 => types::I32,
            Integer::I64 => types::I64,
            Integer::I128 => types::I128,
        },
        Primitive::Float(float) => match float {
            Float::F16 => types::F16,
            Float::F32 => types::F32,
            Float::F64 => types::F64,
            Float::F128 => types::F128,
        },
        // FIXME(erikdesjardins): handle non-default addrspace ptr sizes
        Primitive::Pointer(_) => pointer_ty(tcx),
    }
}

fn clif_type_from_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<types::Type> {
    Some(match ty.kind() {
        ty::Bool => types::I8,
        ty::Uint(size) => match size {
            UintTy::U8 => types::I8,
            UintTy::U16 => types::I16,
            UintTy::U32 => types::I32,
            UintTy::U64 => types::I64,
            UintTy::U128 => types::I128,
            UintTy::Usize => pointer_ty(tcx),
        },
        ty::Int(size) => match size {
            IntTy::I8 => types::I8,
            IntTy::I16 => types::I16,
            IntTy::I32 => types::I32,
            IntTy::I64 => types::I64,
            IntTy::I128 => types::I128,
            IntTy::Isize => pointer_ty(tcx),
        },
        ty::Char => types::I32,
        ty::Float(size) => match size {
            FloatTy::F16 => types::F16,
            FloatTy::F32 => types::F32,
            FloatTy::F64 => types::F64,
            FloatTy::F128 => types::F128,
        },
        ty::FnPtr(..) => pointer_ty(tcx),
        ty::RawPtr(pointee_ty, _) | ty::Ref(_, pointee_ty, _) => {
            if tcx.type_has_metadata(*pointee_ty, ty::TypingEnv::fully_monomorphized()) {
                return None;
            } else {
                pointer_ty(tcx)
            }
        }
        ty::Param(_) => bug!("ty param {:?}", ty),
        _ => return None,
    })
}

fn clif_pair_type_from_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<(types::Type, types::Type)> {
    Some(match ty.kind() {
        ty::Tuple(types) if types.len() == 2 => {
            (clif_type_from_ty(tcx, types[0])?, clif_type_from_ty(tcx, types[1])?)
        }
        ty::RawPtr(pointee_ty, _) | ty::Ref(_, pointee_ty, _) => {
            if tcx.type_has_metadata(*pointee_ty, ty::TypingEnv::fully_monomorphized()) {
                (pointer_ty(tcx), pointer_ty(tcx))
            } else {
                return None;
            }
        }
        _ => return None,
    })
}

pub(crate) fn codegen_icmp_imm(
    fx: &mut FunctionCx<'_, '_, '_>,
    intcc: IntCC,
    lhs: Value,
    rhs: i128,
) -> Value {
    let lhs_ty = fx.bcx.func.dfg.value_type(lhs);
    if lhs_ty == types::I128 {
        // FIXME legalize `icmp_imm.i128` in Cranelift

        let (lhs_lsb, lhs_msb) = fx.bcx.ins().isplit(lhs);
        let (rhs_lsb, rhs_msb) = (rhs as u128 as u64 as i64, (rhs as u128 >> 64) as u64 as i64);

        match intcc {
            IntCC::Equal => {
                let lsb_eq = fx.bcx.ins().icmp_imm(IntCC::Equal, lhs_lsb, rhs_lsb);
                let msb_eq = fx.bcx.ins().icmp_imm(IntCC::Equal, lhs_msb, rhs_msb);
                fx.bcx.ins().band(lsb_eq, msb_eq)
            }
            IntCC::NotEqual => {
                let lsb_ne = fx.bcx.ins().icmp_imm(IntCC::NotEqual, lhs_lsb, rhs_lsb);
                let msb_ne = fx.bcx.ins().icmp_imm(IntCC::NotEqual, lhs_msb, rhs_msb);
                fx.bcx.ins().bor(lsb_ne, msb_ne)
            }
            _ => {
                // if msb_eq {
                //     lsb_cc
                // } else {
                //     msb_cc
                // }

                let msb_eq = fx.bcx.ins().icmp_imm(IntCC::Equal, lhs_msb, rhs_msb);
                let lsb_cc = fx.bcx.ins().icmp_imm(intcc, lhs_lsb, rhs_lsb);
                let msb_cc = fx.bcx.ins().icmp_imm(intcc, lhs_msb, rhs_msb);

                fx.bcx.ins().select(msb_eq, lsb_cc, msb_cc)
            }
        }
    } else {
        let rhs = rhs as i64; // Truncates on purpose in case rhs is actually an unsigned value
        fx.bcx.ins().icmp_imm(intcc, lhs, rhs)
    }
}

pub(crate) fn codegen_bitcast(fx: &mut FunctionCx<'_, '_, '_>, dst_ty: Type, val: Value) -> Value {
    let mut flags = MemFlags::new();
    flags.set_endianness(match fx.tcx.data_layout.endian {
        rustc_abi::Endian::Big => cranelift_codegen::ir::Endianness::Big,
        rustc_abi::Endian::Little => cranelift_codegen::ir::Endianness::Little,
    });
    fx.bcx.ins().bitcast(dst_ty, flags, val)
}

pub(crate) fn type_zero_value(bcx: &mut FunctionBuilder<'_>, ty: Type) -> Value {
    if ty == types::I128 {
        let zero = bcx.ins().iconst(types::I64, 0);
        bcx.ins().iconcat(zero, zero)
    } else {
        bcx.ins().iconst(ty, 0)
    }
}

pub(crate) fn type_min_max_value(
    bcx: &mut FunctionBuilder<'_>,
    ty: Type,
    signed: bool,
) -> (Value, Value) {
    assert!(ty.is_int());

    if ty == types::I128 {
        if signed {
            let min = i128::MIN as u128;
            let min_lsb = bcx.ins().iconst(types::I64, min as u64 as i64);
            let min_msb = bcx.ins().iconst(types::I64, (min >> 64) as u64 as i64);
            let min = bcx.ins().iconcat(min_lsb, min_msb);

            let max = i128::MAX as u128;
            let max_lsb = bcx.ins().iconst(types::I64, max as u64 as i64);
            let max_msb = bcx.ins().iconst(types::I64, (max >> 64) as u64 as i64);
            let max = bcx.ins().iconcat(max_lsb, max_msb);

            return (min, max);
        } else {
            let min_half = bcx.ins().iconst(types::I64, 0);
            let min = bcx.ins().iconcat(min_half, min_half);

            let max_half = bcx.ins().iconst(types::I64, u64::MAX as i64);
            let max = bcx.ins().iconcat(max_half, max_half);

            return (min, max);
        }
    }

    let min = match (ty, signed) {
        (types::I8, false) | (types::I16, false) | (types::I32, false) | (types::I64, false) => {
            0i64
        }
        (types::I8, true) => i64::from(i8::MIN as u8),
        (types::I16, true) => i64::from(i16::MIN as u16),
        (types::I32, true) => i64::from(i32::MIN as u32),
        (types::I64, true) => i64::MIN,
        _ => unreachable!(),
    };

    let max = match (ty, signed) {
        (types::I8, false) => i64::from(u8::MAX),
        (types::I16, false) => i64::from(u16::MAX),
        (types::I32, false) => i64::from(u32::MAX),
        (types::I64, false) => u64::MAX as i64,
        (types::I8, true) => i64::from(i8::MAX as u8),
        (types::I16, true) => i64::from(i16::MAX as u16),
        (types::I32, true) => i64::from(i32::MAX as u32),
        (types::I64, true) => i64::MAX,
        _ => unreachable!(),
    };

    let (min, max) = (bcx.ins().iconst(ty, min), bcx.ins().iconst(ty, max));

    (min, max)
}

pub(crate) fn type_sign(ty: Ty<'_>) -> bool {
    match ty.kind() {
        ty::Ref(..) | ty::RawPtr(..) | ty::FnPtr(..) | ty::Char | ty::Uint(..) | ty::Bool => false,
        ty::Int(..) => true,
        ty::Float(..) => false, // `signed` is unused for floats
        _ => panic!("{}", ty),
    }
}

pub(crate) fn create_wrapper_function(
    module: &mut dyn Module,
    sig: Signature,
    wrapper_name: &str,
    callee_name: &str,
) {
    let wrapper_func_id = module.declare_function(wrapper_name, Linkage::Export, &sig).unwrap();
    let callee_func_id = module.declare_function(callee_name, Linkage::Import, &sig).unwrap();

    let mut ctx = Context::new();
    ctx.func.signature = sig;
    {
        let mut func_ctx = FunctionBuilderContext::new();
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

        let block = bcx.create_block();
        bcx.switch_to_block(block);
        let func = &mut bcx.func.stencil;
        let args = func
            .signature
            .params
            .iter()
            .map(|param| func.dfg.append_block_param(block, param.value_type))
            .collect::<Vec<Value>>();

        let callee_func_ref = module.declare_func_in_func(callee_func_id, &mut bcx.func);
        let call_inst = bcx.ins().call(callee_func_ref, &args);
        let results = bcx.inst_results(call_inst).to_vec(); // Clone to prevent borrow error

        bcx.ins().return_(&results);
        bcx.seal_all_blocks();
        bcx.finalize();
    }
    module.define_function(wrapper_func_id, &mut ctx).unwrap();
}

pub(crate) struct FunctionCx<'m, 'clif, 'tcx: 'm> {
    pub(crate) cx: &'clif mut crate::CodegenCx,
    pub(crate) module: &'m mut dyn Module,
    pub(crate) tcx: TyCtxt<'tcx>,
    pub(crate) target_config: TargetFrontendConfig, // Cached from module
    pub(crate) pointer_type: Type,                  // Cached from module
    pub(crate) constants_cx: ConstantCx,
    pub(crate) func_debug_cx: Option<FunctionDebugContext>,

    pub(crate) instance: Instance<'tcx>,
    pub(crate) symbol_name: String,
    pub(crate) mir: &'tcx Body<'tcx>,
    pub(crate) fn_abi: &'tcx FnAbi<'tcx, Ty<'tcx>>,

    pub(crate) bcx: FunctionBuilder<'clif>,
    pub(crate) block_map: IndexVec<BasicBlock, Block>,
    pub(crate) local_map: IndexVec<Local, CPlace<'tcx>>,

    /// When `#[track_caller]` is used, the implicit caller location is stored in this variable.
    pub(crate) caller_location: Option<CValue<'tcx>>,

    pub(crate) clif_comments: crate::pretty_clif::CommentWriter,

    /// This should only be accessed by `CPlace::new_var`.
    pub(crate) next_ssa_var: u32,
}

impl<'tcx> LayoutOfHelpers<'tcx> for FunctionCx<'_, '_, 'tcx> {
    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        FullyMonomorphizedLayoutCx(self.tcx).handle_layout_err(err, span, ty)
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for FunctionCx<'_, '_, 'tcx> {
    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        FullyMonomorphizedLayoutCx(self.tcx).handle_fn_abi_err(err, span, fn_abi_request)
    }
}

impl<'tcx> layout::HasTyCtxt<'tcx> for FunctionCx<'_, '_, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> rustc_abi::HasDataLayout for FunctionCx<'_, '_, 'tcx> {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'tcx> layout::HasTypingEnv<'tcx> for FunctionCx<'_, '_, 'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        ty::TypingEnv::fully_monomorphized()
    }
}

impl<'tcx> HasTargetSpec for FunctionCx<'_, '_, 'tcx> {
    fn target_spec(&self) -> &Target {
        &self.tcx.sess.target
    }
}

impl<'tcx> FunctionCx<'_, '_, 'tcx> {
    pub(crate) fn monomorphize<T>(&self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>> + Copy,
    {
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            ty::TypingEnv::fully_monomorphized(),
            ty::EarlyBinder::bind(value),
        )
    }

    pub(crate) fn clif_type(&self, ty: Ty<'tcx>) -> Option<Type> {
        clif_type_from_ty(self.tcx, ty)
    }

    pub(crate) fn clif_pair_type(&self, ty: Ty<'tcx>) -> Option<(Type, Type)> {
        clif_pair_type_from_ty(self.tcx, ty)
    }

    pub(crate) fn get_block(&self, bb: BasicBlock) -> Block {
        *self.block_map.get(bb).unwrap()
    }

    pub(crate) fn get_local_place(&mut self, local: Local) -> CPlace<'tcx> {
        *self.local_map.get(local).unwrap_or_else(|| {
            panic!("Local {:?} doesn't exist", local);
        })
    }

    pub(crate) fn create_stack_slot(&mut self, size: u32, align: u32) -> Pointer {
        assert!(
            size % align == 0,
            "size must be a multiple of alignment (size={size}, align={align})"
        );

        let abi_align = if self.tcx.sess.target.arch == "s390x" { 8 } else { 16 };
        if align <= abi_align {
            let stack_slot = self.bcx.create_sized_stack_slot(StackSlotData {
                kind: StackSlotKind::ExplicitSlot,
                // FIXME Don't force the size to a multiple of <abi_align> bytes once Cranelift gets
                // a way to specify stack slot alignment.
                size: (size + abi_align - 1) / abi_align * abi_align,
                align_shift: 4,
            });
            Pointer::stack_slot(stack_slot)
        } else {
            // Alignment is too big to handle using the above hack. Dynamically realign a stack slot
            // instead. This wastes some space for the realignment.
            let stack_slot = self.bcx.create_sized_stack_slot(StackSlotData {
                kind: StackSlotKind::ExplicitSlot,
                // FIXME Don't force the size to a multiple of <abi_align> bytes once Cranelift gets
                // a way to specify stack slot alignment.
                size: (size + align) / abi_align * abi_align,
                align_shift: 4,
            });
            let base_ptr = self.bcx.ins().stack_addr(self.pointer_type, stack_slot, 0);
            let misalign_offset = self.bcx.ins().band_imm(base_ptr, i64::from(align - 1));
            let realign_offset = self.bcx.ins().irsub_imm(misalign_offset, i64::from(align));
            Pointer::new(self.bcx.ins().iadd(base_ptr, realign_offset))
        }
    }

    pub(crate) fn set_debug_loc(&mut self, source_info: mir::SourceInfo) {
        if let Some(debug_context) = &mut self.cx.debug_context {
            let (file_id, line, column) =
                debug_context.get_span_loc(self.tcx, self.mir.span, source_info.span);

            let source_loc =
                self.func_debug_cx.as_mut().unwrap().add_dbg_loc(file_id, line, column);
            self.bcx.set_srcloc(source_loc);
        }
    }

    pub(crate) fn get_caller_location(&mut self, source_info: mir::SourceInfo) -> CValue<'tcx> {
        self.mir.caller_location_span(source_info, self.caller_location, self.tcx, |span| {
            let const_loc = self.tcx.span_as_caller_location(span);
            crate::constant::codegen_const_value(self, const_loc, self.tcx.caller_location_ty())
        })
    }

    pub(crate) fn anonymous_str(&mut self, msg: &str) -> Value {
        let mut data = DataDescription::new();
        data.define(msg.as_bytes().to_vec().into_boxed_slice());
        let msg_id = self.module.declare_anonymous_data(false, false).unwrap();

        // Ignore DuplicateDefinition error, as the data will be the same
        let _ = self.module.define_data(msg_id, &data);

        let local_msg_id = self.module.declare_data_in_func(msg_id, self.bcx.func);
        if self.clif_comments.enabled() {
            self.add_comment(local_msg_id, msg);
        }
        self.bcx.ins().global_value(self.pointer_type, local_msg_id)
    }
}

pub(crate) struct FullyMonomorphizedLayoutCx<'tcx>(pub(crate) TyCtxt<'tcx>);

impl<'tcx> LayoutOfHelpers<'tcx> for FullyMonomorphizedLayoutCx<'tcx> {
    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        if let LayoutError::SizeOverflow(_)
        | LayoutError::InvalidSimd { .. }
        | LayoutError::ReferencesError(_) = err
        {
            self.0.sess.dcx().span_fatal(span, err.to_string())
        } else {
            self.0
                .sess
                .dcx()
                .span_fatal(span, format!("failed to get layout for `{}`: {}", ty, err))
        }
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for FullyMonomorphizedLayoutCx<'tcx> {
    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        if let FnAbiError::Layout(LayoutError::SizeOverflow(_) | LayoutError::InvalidSimd { .. }) =
            err
        {
            self.0.sess.dcx().emit_fatal(Spanned { span, node: err })
        } else {
            match fn_abi_request {
                FnAbiRequest::OfFnPtr { sig, extra_args } => {
                    span_bug!(span, "`fn_abi_of_fn_ptr({sig}, {extra_args:?})` failed: {err:?}");
                }
                FnAbiRequest::OfInstance { instance, extra_args } => {
                    span_bug!(
                        span,
                        "`fn_abi_of_instance({instance}, {extra_args:?})` failed: {err:?}"
                    );
                }
            }
        }
    }
}

impl<'tcx> layout::HasTyCtxt<'tcx> for FullyMonomorphizedLayoutCx<'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.0
    }
}

impl<'tcx> rustc_abi::HasDataLayout for FullyMonomorphizedLayoutCx<'tcx> {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        &self.0.data_layout
    }
}

impl<'tcx> layout::HasTypingEnv<'tcx> for FullyMonomorphizedLayoutCx<'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        ty::TypingEnv::fully_monomorphized()
    }
}

impl<'tcx> HasTargetSpec for FullyMonomorphizedLayoutCx<'tcx> {
    fn target_spec(&self) -> &Target {
        &self.0.sess.target
    }
}
