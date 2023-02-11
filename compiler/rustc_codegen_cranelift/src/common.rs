use cranelift_codegen::isa::TargetFrontendConfig;
use gimli::write::FileId;

use rustc_data_structures::sync::Lrc;
use rustc_index::vec::IndexVec;
use rustc_middle::ty::layout::{
    FnAbiError, FnAbiOfHelpers, FnAbiRequest, LayoutError, LayoutOfHelpers,
};
use rustc_span::SourceFile;
use rustc_target::abi::call::FnAbi;
use rustc_target::abi::{Integer, Primitive};
use rustc_target::spec::{HasTargetSpec, Target};

use crate::constant::ConstantCx;
use crate::debuginfo::FunctionDebugContext;
use crate::prelude::*;

pub(crate) fn pointer_ty(tcx: TyCtxt<'_>) -> types::Type {
    match tcx.data_layout.pointer_size.bits() {
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
        Primitive::F32 => types::F32,
        Primitive::F64 => types::F64,
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
            FloatTy::F32 => types::F32,
            FloatTy::F64 => types::F64,
        },
        ty::FnPtr(_) => pointer_ty(tcx),
        ty::RawPtr(TypeAndMut { ty: pointee_ty, mutbl: _ }) | ty::Ref(_, pointee_ty, _) => {
            if has_ptr_meta(tcx, *pointee_ty) {
                return None;
            } else {
                pointer_ty(tcx)
            }
        }
        ty::Adt(adt_def, _) if adt_def.repr().simd() => {
            let (element, count) = match &tcx.layout_of(ParamEnv::reveal_all().and(ty)).unwrap().abi
            {
                Abi::Vector { element, count } => (element.clone(), *count),
                _ => unreachable!(),
            };

            match scalar_to_clif_type(tcx, element).by(u32::try_from(count).unwrap()) {
                // Cranelift currently only implements icmp for 128bit vectors.
                Some(vector_ty) if vector_ty.bits() == 128 => vector_ty,
                _ => return None,
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
            let a = clif_type_from_ty(tcx, types[0])?;
            let b = clif_type_from_ty(tcx, types[1])?;
            if a.is_vector() || b.is_vector() {
                return None;
            }
            (a, b)
        }
        ty::RawPtr(TypeAndMut { ty: pointee_ty, mutbl: _ }) | ty::Ref(_, pointee_ty, _) => {
            if has_ptr_meta(tcx, *pointee_ty) {
                (pointer_ty(tcx), pointer_ty(tcx))
            } else {
                return None;
            }
        }
        _ => return None,
    })
}

/// Is a pointer to this type a fat ptr?
pub(crate) fn has_ptr_meta<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    let ptr_ty = tcx.mk_ptr(TypeAndMut { ty, mutbl: rustc_hir::Mutability::Not });
    match &tcx.layout_of(ParamEnv::reveal_all().and(ptr_ty)).unwrap().abi {
        Abi::Scalar(_) => false,
        Abi::ScalarPair(_, _) => true,
        abi => unreachable!("Abi of ptr to {:?} is {:?}???", ty, abi),
    }
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
        rustc_target::abi::Endian::Big => cranelift_codegen::ir::Endianness::Big,
        rustc_target::abi::Endian::Little => cranelift_codegen::ir::Endianness::Little,
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
        (types::I8, true) => i64::from(i8::MIN),
        (types::I16, true) => i64::from(i16::MIN),
        (types::I32, true) => i64::from(i32::MIN),
        (types::I64, true) => i64::MIN,
        _ => unreachable!(),
    };

    let max = match (ty, signed) {
        (types::I8, false) => i64::from(u8::MAX),
        (types::I16, false) => i64::from(u16::MAX),
        (types::I32, false) => i64::from(u32::MAX),
        (types::I64, false) => u64::MAX as i64,
        (types::I8, true) => i64::from(i8::MAX),
        (types::I16, true) => i64::from(i16::MAX),
        (types::I32, true) => i64::from(i32::MAX),
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
    unwind_context: &mut UnwindContext,
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
    unwind_context.add_function(wrapper_func_id, &ctx, module.isa());
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
    pub(crate) fn_abi: Option<&'tcx FnAbi<'tcx, Ty<'tcx>>>,

    pub(crate) bcx: FunctionBuilder<'clif>,
    pub(crate) block_map: IndexVec<BasicBlock, Block>,
    pub(crate) local_map: IndexVec<Local, CPlace<'tcx>>,

    /// When `#[track_caller]` is used, the implicit caller location is stored in this variable.
    pub(crate) caller_location: Option<CValue<'tcx>>,

    pub(crate) clif_comments: crate::pretty_clif::CommentWriter,

    /// Last accessed source file and it's debuginfo file id.
    ///
    /// For optimization purposes only
    pub(crate) last_source_file: Option<(Lrc<SourceFile>, FileId)>,

    /// This should only be accessed by `CPlace::new_var`.
    pub(crate) next_ssa_var: u32,
}

impl<'tcx> LayoutOfHelpers<'tcx> for FunctionCx<'_, '_, 'tcx> {
    type LayoutOfResult = TyAndLayout<'tcx>;

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        RevealAllLayoutCx(self.tcx).handle_layout_err(err, span, ty)
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for FunctionCx<'_, '_, 'tcx> {
    type FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>;

    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        RevealAllLayoutCx(self.tcx).handle_fn_abi_err(err, span, fn_abi_request)
    }
}

impl<'tcx> layout::HasTyCtxt<'tcx> for FunctionCx<'_, '_, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> rustc_target::abi::HasDataLayout for FunctionCx<'_, '_, 'tcx> {
    fn data_layout(&self) -> &rustc_target::abi::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'tcx> layout::HasParamEnv<'tcx> for FunctionCx<'_, '_, 'tcx> {
    fn param_env(&self) -> ParamEnv<'tcx> {
        ParamEnv::reveal_all()
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
        T: TypeFoldable<'tcx> + Copy,
    {
        self.instance.subst_mir_and_normalize_erasing_regions(
            self.tcx,
            ty::ParamEnv::reveal_all(),
            value,
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

    pub(crate) fn set_debug_loc(&mut self, source_info: mir::SourceInfo) {
        if let Some(debug_context) = &mut self.cx.debug_context {
            let (file, line, column) =
                DebugContext::get_span_loc(self.tcx, self.mir.span, source_info.span);

            // add_source_file is very slow.
            // Optimize for the common case of the current file not being changed.
            let mut cached_file_id = None;
            if let Some((ref last_source_file, last_file_id)) = self.last_source_file {
                // If the allocations are not equal, the files may still be equal, but that
                // doesn't matter, as this is just an optimization.
                if rustc_data_structures::sync::Lrc::ptr_eq(last_source_file, &file) {
                    cached_file_id = Some(last_file_id);
                }
            }

            let file_id = if let Some(file_id) = cached_file_id {
                file_id
            } else {
                debug_context.add_source_file(&file)
            };

            let source_loc =
                self.func_debug_cx.as_mut().unwrap().add_dbg_loc(file_id, line, column);
            self.bcx.set_srcloc(source_loc);
        }
    }

    // Note: must be kept in sync with get_caller_location from cg_ssa
    pub(crate) fn get_caller_location(&mut self, mut source_info: mir::SourceInfo) -> CValue<'tcx> {
        let span_to_caller_location = |fx: &mut FunctionCx<'_, '_, 'tcx>, span: Span| {
            let topmost = span.ctxt().outer_expn().expansion_cause().unwrap_or(span);
            let caller = fx.tcx.sess.source_map().lookup_char_pos(topmost.lo());
            let const_loc = fx.tcx.const_caller_location((
                rustc_span::symbol::Symbol::intern(
                    &caller.file.name.prefer_remapped().to_string_lossy(),
                ),
                caller.line as u32,
                caller.col_display as u32 + 1,
            ));
            crate::constant::codegen_const_value(fx, const_loc, fx.tcx.caller_location_ty())
        };

        // Walk up the `SourceScope`s, in case some of them are from MIR inlining.
        // If so, the starting `source_info.span` is in the innermost inlined
        // function, and will be replaced with outer callsite spans as long
        // as the inlined functions were `#[track_caller]`.
        loop {
            let scope_data = &self.mir.source_scopes[source_info.scope];

            if let Some((callee, callsite_span)) = scope_data.inlined {
                // Stop inside the most nested non-`#[track_caller]` function,
                // before ever reaching its caller (which is irrelevant).
                if !callee.def.requires_caller_location(self.tcx) {
                    return span_to_caller_location(self, source_info.span);
                }
                source_info.span = callsite_span;
            }

            // Skip past all of the parents with `inlined: None`.
            match scope_data.inlined_parent_scope {
                Some(parent) => source_info.scope = parent,
                None => break,
            }
        }

        // No inlined `SourceScope`s, or all of them were `#[track_caller]`.
        self.caller_location.unwrap_or_else(|| span_to_caller_location(self, source_info.span))
    }

    pub(crate) fn anonymous_str(&mut self, msg: &str) -> Value {
        let mut data_ctx = DataContext::new();
        data_ctx.define(msg.as_bytes().to_vec().into_boxed_slice());
        let msg_id = self.module.declare_anonymous_data(false, false).unwrap();

        // Ignore DuplicateDefinition error, as the data will be the same
        let _ = self.module.define_data(msg_id, &data_ctx);

        let local_msg_id = self.module.declare_data_in_func(msg_id, self.bcx.func);
        if self.clif_comments.enabled() {
            self.add_comment(local_msg_id, msg);
        }
        self.bcx.ins().global_value(self.pointer_type, local_msg_id)
    }
}

pub(crate) struct RevealAllLayoutCx<'tcx>(pub(crate) TyCtxt<'tcx>);

impl<'tcx> LayoutOfHelpers<'tcx> for RevealAllLayoutCx<'tcx> {
    type LayoutOfResult = TyAndLayout<'tcx>;

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        if let layout::LayoutError::SizeOverflow(_) = err {
            self.0.sess.span_fatal(span, &err.to_string())
        } else {
            span_bug!(span, "failed to get layout for `{}`: {}", ty, err)
        }
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for RevealAllLayoutCx<'tcx> {
    type FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>;

    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        if let FnAbiError::Layout(LayoutError::SizeOverflow(_)) = err {
            self.0.sess.span_fatal(span, &err.to_string())
        } else {
            match fn_abi_request {
                FnAbiRequest::OfFnPtr { sig, extra_args } => {
                    span_bug!(
                        span,
                        "`fn_abi_of_fn_ptr({}, {:?})` failed: {}",
                        sig,
                        extra_args,
                        err
                    );
                }
                FnAbiRequest::OfInstance { instance, extra_args } => {
                    span_bug!(
                        span,
                        "`fn_abi_of_instance({}, {:?})` failed: {}",
                        instance,
                        extra_args,
                        err
                    );
                }
            }
        }
    }
}

impl<'tcx> layout::HasTyCtxt<'tcx> for RevealAllLayoutCx<'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.0
    }
}

impl<'tcx> rustc_target::abi::HasDataLayout for RevealAllLayoutCx<'tcx> {
    fn data_layout(&self) -> &rustc_target::abi::TargetDataLayout {
        &self.0.data_layout
    }
}

impl<'tcx> layout::HasParamEnv<'tcx> for RevealAllLayoutCx<'tcx> {
    fn param_env(&self) -> ParamEnv<'tcx> {
        ParamEnv::reveal_all()
    }
}

impl<'tcx> HasTargetSpec for RevealAllLayoutCx<'tcx> {
    fn target_spec(&self) -> &Target {
        &self.0.sess.target
    }
}
