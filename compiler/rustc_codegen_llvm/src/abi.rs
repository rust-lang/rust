use crate::attributes;
use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::llvm::{self, Attribute, AttributePlace};
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;

use rustc_codegen_ssa::mir::operand::OperandValue;
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::MemFlags;
use rustc_middle::bug;
use rustc_middle::ty::layout::LayoutOf;
pub use rustc_middle::ty::layout::{FAT_PTR_ADDR, FAT_PTR_EXTRA};
use rustc_middle::ty::Ty;
use rustc_session::config;
use rustc_target::abi::call::ArgAbi;
pub use rustc_target::abi::call::*;
use rustc_target::abi::{self, HasDataLayout, Int};
pub use rustc_target::spec::abi::Abi;
use rustc_target::spec::SanitizerSet;

use libc::c_uint;
use smallvec::SmallVec;

pub trait ArgAttributesExt {
    fn apply_attrs_to_llfn(&self, idx: AttributePlace, cx: &CodegenCx<'_, '_>, llfn: &Value);
    fn apply_attrs_to_callsite(
        &self,
        idx: AttributePlace,
        cx: &CodegenCx<'_, '_>,
        callsite: &Value,
    );
}

fn should_use_mutable_noalias(cx: &CodegenCx<'_, '_>) -> bool {
    // LLVM prior to version 12 had known miscompiles in the presence of
    // noalias attributes (see #54878), but we don't support earlier
    // versions at all anymore. We now enable mutable noalias by default.
    cx.tcx.sess.opts.unstable_opts.mutable_noalias.unwrap_or(true)
}

const ABI_AFFECTING_ATTRIBUTES: [(ArgAttribute, llvm::AttributeKind); 1] =
    [(ArgAttribute::InReg, llvm::AttributeKind::InReg)];

const OPTIMIZATION_ATTRIBUTES: [(ArgAttribute, llvm::AttributeKind); 5] = [
    (ArgAttribute::NoAlias, llvm::AttributeKind::NoAlias),
    (ArgAttribute::NoCapture, llvm::AttributeKind::NoCapture),
    (ArgAttribute::NonNull, llvm::AttributeKind::NonNull),
    (ArgAttribute::ReadOnly, llvm::AttributeKind::ReadOnly),
    (ArgAttribute::NoUndef, llvm::AttributeKind::NoUndef),
];

fn get_attrs<'ll>(this: &ArgAttributes, cx: &CodegenCx<'ll, '_>) -> SmallVec<[&'ll Attribute; 8]> {
    let mut regular = this.regular;

    let mut attrs = SmallVec::new();

    // ABI-affecting attributes must always be applied
    for (attr, llattr) in ABI_AFFECTING_ATTRIBUTES {
        if regular.contains(attr) {
            attrs.push(llattr.create_attr(cx.llcx));
        }
    }
    if let Some(align) = this.pointee_align {
        attrs.push(llvm::CreateAlignmentAttr(cx.llcx, align.bytes()));
    }
    match this.arg_ext {
        ArgExtension::None => {}
        ArgExtension::Zext => attrs.push(llvm::AttributeKind::ZExt.create_attr(cx.llcx)),
        ArgExtension::Sext => attrs.push(llvm::AttributeKind::SExt.create_attr(cx.llcx)),
    }

    // Only apply remaining attributes when optimizing
    if cx.sess().opts.optimize != config::OptLevel::No {
        let deref = this.pointee_size.bytes();
        if deref != 0 {
            if regular.contains(ArgAttribute::NonNull) {
                attrs.push(llvm::CreateDereferenceableAttr(cx.llcx, deref));
            } else {
                attrs.push(llvm::CreateDereferenceableOrNullAttr(cx.llcx, deref));
            }
            regular -= ArgAttribute::NonNull;
        }
        for (attr, llattr) in OPTIMIZATION_ATTRIBUTES {
            if regular.contains(attr) {
                attrs.push(llattr.create_attr(cx.llcx));
            }
        }
        if regular.contains(ArgAttribute::NoAliasMutRef) && should_use_mutable_noalias(cx) {
            attrs.push(llvm::AttributeKind::NoAlias.create_attr(cx.llcx));
        }
    } else if cx.tcx.sess.opts.unstable_opts.sanitizer.contains(SanitizerSet::MEMORY) {
        // If we're not optimising, *but* memory sanitizer is on, emit noundef, since it affects
        // memory sanitizer's behavior.

        if regular.contains(ArgAttribute::NoUndef) {
            attrs.push(llvm::AttributeKind::NoUndef.create_attr(cx.llcx));
        }
    }

    attrs
}

impl ArgAttributesExt for ArgAttributes {
    fn apply_attrs_to_llfn(&self, idx: AttributePlace, cx: &CodegenCx<'_, '_>, llfn: &Value) {
        let attrs = get_attrs(self, cx);
        attributes::apply_to_llfn(llfn, idx, &attrs);
    }

    fn apply_attrs_to_callsite(
        &self,
        idx: AttributePlace,
        cx: &CodegenCx<'_, '_>,
        callsite: &Value,
    ) {
        let attrs = get_attrs(self, cx);
        attributes::apply_to_callsite(callsite, idx, &attrs);
    }
}

pub trait LlvmType {
    fn llvm_type<'ll>(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type;
}

impl LlvmType for Reg {
    fn llvm_type<'ll>(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type {
        match self.kind {
            RegKind::Integer => cx.type_ix(self.size.bits()),
            RegKind::Float => match self.size.bits() {
                32 => cx.type_f32(),
                64 => cx.type_f64(),
                _ => bug!("unsupported float: {:?}", self),
            },
            RegKind::Vector => cx.type_vector(cx.type_i8(), self.size.bytes()),
        }
    }
}

impl LlvmType for CastTarget {
    fn llvm_type<'ll>(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type {
        let rest_ll_unit = self.rest.unit.llvm_type(cx);
        let (rest_count, rem_bytes) = if self.rest.unit.size.bytes() == 0 {
            (0, 0)
        } else {
            (
                self.rest.total.bytes() / self.rest.unit.size.bytes(),
                self.rest.total.bytes() % self.rest.unit.size.bytes(),
            )
        };

        if self.prefix.iter().all(|x| x.is_none()) {
            // Simplify to a single unit when there is no prefix and size <= unit size
            if self.rest.total <= self.rest.unit.size {
                return rest_ll_unit;
            }

            // Simplify to array when all chunks are the same size and type
            if rem_bytes == 0 {
                return cx.type_array(rest_ll_unit, rest_count);
            }
        }

        // Create list of fields in the main structure
        let mut args: Vec<_> = self
            .prefix
            .iter()
            .flat_map(|option_reg| option_reg.map(|reg| reg.llvm_type(cx)))
            .chain((0..rest_count).map(|_| rest_ll_unit))
            .collect();

        // Append final integer
        if rem_bytes != 0 {
            // Only integers can be really split further.
            assert_eq!(self.rest.unit.kind, RegKind::Integer);
            args.push(cx.type_ix(rem_bytes * 8));
        }

        cx.type_struct(&args, false)
    }
}

pub trait ArgAbiExt<'ll, 'tcx> {
    fn memory_ty(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn store(
        &self,
        bx: &mut Builder<'_, 'll, 'tcx>,
        val: &'ll Value,
        dst: PlaceRef<'tcx, &'ll Value>,
    );
    fn store_fn_arg(
        &self,
        bx: &mut Builder<'_, 'll, 'tcx>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, &'ll Value>,
    );
}

impl<'ll, 'tcx> ArgAbiExt<'ll, 'tcx> for ArgAbi<'tcx, Ty<'tcx>> {
    /// Gets the LLVM type for a place of the original Rust type of
    /// this argument/return, i.e., the result of `type_of::type_of`.
    fn memory_ty(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type {
        self.layout.llvm_type(cx)
    }

    /// Stores a direct/indirect value described by this ArgAbi into a
    /// place for the original Rust type of this argument/return.
    /// Can be used for both storing formal arguments into Rust variables
    /// or results of call/invoke instructions into their destinations.
    fn store(
        &self,
        bx: &mut Builder<'_, 'll, 'tcx>,
        val: &'ll Value,
        dst: PlaceRef<'tcx, &'ll Value>,
    ) {
        if self.is_ignore() {
            return;
        }
        if self.is_sized_indirect() {
            OperandValue::Ref(val, None, self.layout.align.abi).store(bx, dst)
        } else if self.is_unsized_indirect() {
            bug!("unsized `ArgAbi` must be handled through `store_fn_arg`");
        } else if let PassMode::Cast(cast, _) = &self.mode {
            // FIXME(eddyb): Figure out when the simpler Store is safe, clang
            // uses it for i16 -> {i8, i8}, but not for i24 -> {i8, i8, i8}.
            let can_store_through_cast_ptr = false;
            if can_store_through_cast_ptr {
                let cast_ptr_llty = bx.type_ptr_to(cast.llvm_type(bx));
                let cast_dst = bx.pointercast(dst.llval, cast_ptr_llty);
                bx.store(val, cast_dst, self.layout.align.abi);
            } else {
                // The actual return type is a struct, but the ABI
                // adaptation code has cast it into some scalar type.  The
                // code that follows is the only reliable way I have
                // found to do a transform like i64 -> {i32,i32}.
                // Basically we dump the data onto the stack then memcpy it.
                //
                // Other approaches I tried:
                // - Casting rust ret pointer to the foreign type and using Store
                //   is (a) unsafe if size of foreign type > size of rust type and
                //   (b) runs afoul of strict aliasing rules, yielding invalid
                //   assembly under -O (specifically, the store gets removed).
                // - Truncating foreign type to correct integral type and then
                //   bitcasting to the struct type yields invalid cast errors.

                // We instead thus allocate some scratch space...
                let scratch_size = cast.size(bx);
                let scratch_align = cast.align(bx);
                let llscratch = bx.alloca(cast.llvm_type(bx), scratch_align);
                bx.lifetime_start(llscratch, scratch_size);

                // ... where we first store the value...
                bx.store(val, llscratch, scratch_align);

                // ... and then memcpy it to the intended destination.
                bx.memcpy(
                    dst.llval,
                    self.layout.align.abi,
                    llscratch,
                    scratch_align,
                    bx.const_usize(self.layout.size.bytes()),
                    MemFlags::empty(),
                );

                bx.lifetime_end(llscratch, scratch_size);
            }
        } else {
            OperandValue::Immediate(val).store(bx, dst);
        }
    }

    fn store_fn_arg(
        &self,
        bx: &mut Builder<'_, 'll, 'tcx>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, &'ll Value>,
    ) {
        let mut next = || {
            let val = llvm::get_param(bx.llfn(), *idx as c_uint);
            *idx += 1;
            val
        };
        match self.mode {
            PassMode::Ignore => {}
            PassMode::Pair(..) => {
                OperandValue::Pair(next(), next()).store(bx, dst);
            }
            PassMode::Indirect { attrs: _, extra_attrs: Some(_), on_stack: _ } => {
                OperandValue::Ref(next(), Some(next()), self.layout.align.abi).store(bx, dst);
            }
            PassMode::Direct(_)
            | PassMode::Indirect { attrs: _, extra_attrs: None, on_stack: _ }
            | PassMode::Cast(..) => {
                let next_arg = next();
                self.store(bx, next_arg, dst);
            }
        }
    }
}

impl<'ll, 'tcx> ArgAbiMethods<'tcx> for Builder<'_, 'll, 'tcx> {
    fn store_fn_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, Self::Value>,
    ) {
        arg_abi.store_fn_arg(self, idx, dst)
    }
    fn store_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        val: &'ll Value,
        dst: PlaceRef<'tcx, &'ll Value>,
    ) {
        arg_abi.store(self, val, dst)
    }
    fn arg_memory_ty(&self, arg_abi: &ArgAbi<'tcx, Ty<'tcx>>) -> &'ll Type {
        arg_abi.memory_ty(self)
    }
}

pub trait FnAbiLlvmExt<'ll, 'tcx> {
    fn llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn ptr_to_llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn llvm_cconv(&self) -> llvm::CallConv;
    fn apply_attrs_llfn(&self, cx: &CodegenCx<'ll, 'tcx>, llfn: &'ll Value);
    fn apply_attrs_callsite(&self, bx: &mut Builder<'_, 'll, 'tcx>, callsite: &'ll Value);
}

impl<'ll, 'tcx> FnAbiLlvmExt<'ll, 'tcx> for FnAbi<'tcx, Ty<'tcx>> {
    fn llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type {
        // Ignore "extra" args from the call site for C variadic functions.
        // Only the "fixed" args are part of the LLVM function signature.
        let args =
            if self.c_variadic { &self.args[..self.fixed_count as usize] } else { &self.args };

        // This capacity calculation is approximate.
        let mut llargument_tys = Vec::with_capacity(
            self.args.len() + if let PassMode::Indirect { .. } = self.ret.mode { 1 } else { 0 },
        );

        let llreturn_ty = match &self.ret.mode {
            PassMode::Ignore => cx.type_void(),
            PassMode::Direct(_) | PassMode::Pair(..) => self.ret.layout.immediate_llvm_type(cx),
            PassMode::Cast(cast, _) => cast.llvm_type(cx),
            PassMode::Indirect { .. } => {
                llargument_tys.push(cx.type_ptr_to(self.ret.memory_ty(cx)));
                cx.type_void()
            }
        };

        for arg in args {
            let llarg_ty = match &arg.mode {
                PassMode::Ignore => continue,
                PassMode::Direct(_) => arg.layout.immediate_llvm_type(cx),
                PassMode::Pair(..) => {
                    llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 0, true));
                    llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 1, true));
                    continue;
                }
                PassMode::Indirect { attrs: _, extra_attrs: Some(_), on_stack: _ } => {
                    let ptr_ty = cx.tcx.mk_mut_ptr(arg.layout.ty);
                    let ptr_layout = cx.layout_of(ptr_ty);
                    llargument_tys.push(ptr_layout.scalar_pair_element_llvm_type(cx, 0, true));
                    llargument_tys.push(ptr_layout.scalar_pair_element_llvm_type(cx, 1, true));
                    continue;
                }
                PassMode::Cast(cast, pad_i32) => {
                    // add padding
                    if *pad_i32 {
                        llargument_tys.push(Reg::i32().llvm_type(cx));
                    }
                    cast.llvm_type(cx)
                }
                PassMode::Indirect { attrs: _, extra_attrs: None, on_stack: _ } => {
                    cx.type_ptr_to(arg.memory_ty(cx))
                }
            };
            llargument_tys.push(llarg_ty);
        }

        if self.c_variadic {
            cx.type_variadic_func(&llargument_tys, llreturn_ty)
        } else {
            cx.type_func(&llargument_tys, llreturn_ty)
        }
    }

    fn ptr_to_llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type {
        unsafe {
            llvm::LLVMPointerType(
                self.llvm_type(cx),
                cx.data_layout().instruction_address_space.0 as c_uint,
            )
        }
    }

    fn llvm_cconv(&self) -> llvm::CallConv {
        match self.conv {
            Conv::C | Conv::Rust | Conv::CCmseNonSecureCall => llvm::CCallConv,
            Conv::RustCold => llvm::ColdCallConv,
            Conv::AmdGpuKernel => llvm::AmdGpuKernel,
            Conv::AvrInterrupt => llvm::AvrInterrupt,
            Conv::AvrNonBlockingInterrupt => llvm::AvrNonBlockingInterrupt,
            Conv::ArmAapcs => llvm::ArmAapcsCallConv,
            Conv::Msp430Intr => llvm::Msp430Intr,
            Conv::PtxKernel => llvm::PtxKernel,
            Conv::X86Fastcall => llvm::X86FastcallCallConv,
            Conv::X86Intr => llvm::X86_Intr,
            Conv::X86Stdcall => llvm::X86StdcallCallConv,
            Conv::X86ThisCall => llvm::X86_ThisCall,
            Conv::X86VectorCall => llvm::X86_VectorCall,
            Conv::X86_64SysV => llvm::X86_64_SysV,
            Conv::X86_64Win64 => llvm::X86_64_Win64,
        }
    }

    fn apply_attrs_llfn(&self, cx: &CodegenCx<'ll, 'tcx>, llfn: &'ll Value) {
        let mut func_attrs = SmallVec::<[_; 2]>::new();
        if self.ret.layout.abi.is_uninhabited() {
            func_attrs.push(llvm::AttributeKind::NoReturn.create_attr(cx.llcx));
        }
        if !self.can_unwind {
            func_attrs.push(llvm::AttributeKind::NoUnwind.create_attr(cx.llcx));
        }
        attributes::apply_to_llfn(llfn, llvm::AttributePlace::Function, &{ func_attrs });

        let mut i = 0;
        let mut apply = |attrs: &ArgAttributes| {
            attrs.apply_attrs_to_llfn(llvm::AttributePlace::Argument(i), cx, llfn);
            i += 1;
            i - 1
        };
        match &self.ret.mode {
            PassMode::Direct(attrs) => {
                attrs.apply_attrs_to_llfn(llvm::AttributePlace::ReturnValue, cx, llfn);
            }
            PassMode::Indirect { attrs, extra_attrs: _, on_stack } => {
                assert!(!on_stack);
                let i = apply(attrs);
                let sret = llvm::CreateStructRetAttr(cx.llcx, self.ret.layout.llvm_type(cx));
                attributes::apply_to_llfn(llfn, llvm::AttributePlace::Argument(i), &[sret]);
            }
            PassMode::Cast(cast, _) => {
                cast.attrs.apply_attrs_to_llfn(llvm::AttributePlace::ReturnValue, cx, llfn);
            }
            _ => {}
        }
        for arg in self.args.iter() {
            match &arg.mode {
                PassMode::Ignore => {}
                PassMode::Indirect { attrs, extra_attrs: None, on_stack: true } => {
                    let i = apply(attrs);
                    let byval = llvm::CreateByValAttr(cx.llcx, arg.layout.llvm_type(cx));
                    attributes::apply_to_llfn(llfn, llvm::AttributePlace::Argument(i), &[byval]);
                }
                PassMode::Direct(attrs)
                | PassMode::Indirect { attrs, extra_attrs: None, on_stack: false } => {
                    apply(attrs);
                }
                PassMode::Indirect { attrs, extra_attrs: Some(extra_attrs), on_stack } => {
                    assert!(!on_stack);
                    apply(attrs);
                    apply(extra_attrs);
                }
                PassMode::Pair(a, b) => {
                    apply(a);
                    apply(b);
                }
                PassMode::Cast(cast, pad_i32) => {
                    if *pad_i32 {
                        apply(&ArgAttributes::new());
                    }
                    apply(&cast.attrs);
                }
            }
        }
    }

    fn apply_attrs_callsite(&self, bx: &mut Builder<'_, 'll, 'tcx>, callsite: &'ll Value) {
        let mut func_attrs = SmallVec::<[_; 2]>::new();
        if self.ret.layout.abi.is_uninhabited() {
            func_attrs.push(llvm::AttributeKind::NoReturn.create_attr(bx.cx.llcx));
        }
        if !self.can_unwind {
            func_attrs.push(llvm::AttributeKind::NoUnwind.create_attr(bx.cx.llcx));
        }
        attributes::apply_to_callsite(callsite, llvm::AttributePlace::Function, &{ func_attrs });

        let mut i = 0;
        let mut apply = |cx: &CodegenCx<'_, '_>, attrs: &ArgAttributes| {
            attrs.apply_attrs_to_callsite(llvm::AttributePlace::Argument(i), cx, callsite);
            i += 1;
            i - 1
        };
        match &self.ret.mode {
            PassMode::Direct(attrs) => {
                attrs.apply_attrs_to_callsite(llvm::AttributePlace::ReturnValue, bx.cx, callsite);
            }
            PassMode::Indirect { attrs, extra_attrs: _, on_stack } => {
                assert!(!on_stack);
                let i = apply(bx.cx, attrs);
                let sret = llvm::CreateStructRetAttr(bx.cx.llcx, self.ret.layout.llvm_type(bx));
                attributes::apply_to_callsite(callsite, llvm::AttributePlace::Argument(i), &[sret]);
            }
            PassMode::Cast(cast, _) => {
                cast.attrs.apply_attrs_to_callsite(
                    llvm::AttributePlace::ReturnValue,
                    &bx.cx,
                    callsite,
                );
            }
            _ => {}
        }
        if let abi::Abi::Scalar(scalar) = self.ret.layout.abi {
            // If the value is a boolean, the range is 0..2 and that ultimately
            // become 0..0 when the type becomes i1, which would be rejected
            // by the LLVM verifier.
            if let Int(..) = scalar.primitive() {
                if !scalar.is_bool() && !scalar.is_always_valid(bx) {
                    bx.range_metadata(callsite, scalar.valid_range(bx));
                }
            }
        }
        for arg in self.args.iter() {
            match &arg.mode {
                PassMode::Ignore => {}
                PassMode::Indirect { attrs, extra_attrs: None, on_stack: true } => {
                    let i = apply(bx.cx, attrs);
                    let byval = llvm::CreateByValAttr(bx.cx.llcx, arg.layout.llvm_type(bx));
                    attributes::apply_to_callsite(
                        callsite,
                        llvm::AttributePlace::Argument(i),
                        &[byval],
                    );
                }
                PassMode::Direct(attrs)
                | PassMode::Indirect { attrs, extra_attrs: None, on_stack: false } => {
                    apply(bx.cx, attrs);
                }
                PassMode::Indirect { attrs, extra_attrs: Some(extra_attrs), on_stack: _ } => {
                    apply(bx.cx, attrs);
                    apply(bx.cx, extra_attrs);
                }
                PassMode::Pair(a, b) => {
                    apply(bx.cx, a);
                    apply(bx.cx, b);
                }
                PassMode::Cast(cast, pad_i32) => {
                    if *pad_i32 {
                        apply(bx.cx, &ArgAttributes::new());
                    }
                    apply(bx.cx, &cast.attrs);
                }
            }
        }

        let cconv = self.llvm_cconv();
        if cconv != llvm::CCallConv {
            llvm::SetInstructionCallConv(callsite, cconv);
        }

        if self.conv == Conv::CCmseNonSecureCall {
            // This will probably get ignored on all targets but those supporting the TrustZone-M
            // extension (thumbv8m targets).
            let cmse_nonsecure_call = llvm::CreateAttrString(bx.cx.llcx, "cmse_nonsecure_call");
            attributes::apply_to_callsite(
                callsite,
                llvm::AttributePlace::Function,
                &[cmse_nonsecure_call],
            );
        }

        // Some intrinsics require that an elementtype attribute (with the pointee type of a
        // pointer argument) is added to the callsite.
        let element_type_index = unsafe { llvm::LLVMRustGetElementTypeArgIndex(callsite) };
        if element_type_index >= 0 {
            let arg_ty = self.args[element_type_index as usize].layout.ty;
            let pointee_ty = arg_ty.builtin_deref(true).expect("Must be pointer argument").ty;
            let element_type_attr = unsafe {
                llvm::LLVMRustCreateElementTypeAttr(bx.llcx, bx.layout_of(pointee_ty).llvm_type(bx))
            };
            attributes::apply_to_callsite(
                callsite,
                llvm::AttributePlace::Argument(element_type_index as u32),
                &[element_type_attr],
            );
        }
    }
}

impl<'tcx> AbiBuilderMethods<'tcx> for Builder<'_, '_, 'tcx> {
    fn apply_attrs_callsite(&mut self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>, callsite: Self::Value) {
        fn_abi.apply_attrs_callsite(self, callsite)
    }

    fn get_param(&mut self, index: usize) -> Self::Value {
        llvm::get_param(self.llfn(), index as c_uint)
    }
}
