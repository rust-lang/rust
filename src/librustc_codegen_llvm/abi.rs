use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::llvm::{self, AttributePlace};
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;

use rustc_codegen_ssa::mir::operand::OperandValue;
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::MemFlags;
use rustc_middle::bug;
use rustc_middle::ty::layout::{self};
use rustc_middle::ty::Ty;
use rustc_target::abi::call::ArgAbi;
use rustc_target::abi::{HasDataLayout, LayoutOf};

use libc::c_uint;

pub use rustc_middle::ty::layout::{FAT_PTR_ADDR, FAT_PTR_EXTRA};
pub use rustc_target::abi::call::*;
pub use rustc_target::spec::abi::Abi;

macro_rules! for_each_kind {
    ($flags: ident, $f: ident, $($kind: ident),+) => ({
        $(if $flags.contains(ArgAttribute::$kind) { $f(llvm::Attribute::$kind) })+
    })
}

trait ArgAttributeExt {
    fn for_each_kind<F>(&self, f: F)
    where
        F: FnMut(llvm::Attribute);
}

impl ArgAttributeExt for ArgAttribute {
    fn for_each_kind<F>(&self, mut f: F)
    where
        F: FnMut(llvm::Attribute),
    {
        for_each_kind!(self, f, NoAlias, NoCapture, NonNull, ReadOnly, SExt, StructRet, ZExt, InReg)
    }
}

pub trait ArgAttributesExt {
    fn apply_llfn(&self, idx: AttributePlace, llfn: &Value, ty: Option<&Type>);
    fn apply_callsite(&self, idx: AttributePlace, callsite: &Value, ty: Option<&Type>);
}

impl ArgAttributesExt for ArgAttributes {
    fn apply_llfn(&self, idx: AttributePlace, llfn: &Value, ty: Option<&Type>) {
        let mut regular = self.regular;
        unsafe {
            let deref = self.pointee_size.bytes();
            if deref != 0 {
                if regular.contains(ArgAttribute::NonNull) {
                    llvm::LLVMRustAddDereferenceableAttr(llfn, idx.as_uint(), deref);
                } else {
                    llvm::LLVMRustAddDereferenceableOrNullAttr(llfn, idx.as_uint(), deref);
                }
                regular -= ArgAttribute::NonNull;
            }
            if let Some(align) = self.pointee_align {
                llvm::LLVMRustAddAlignmentAttr(llfn, idx.as_uint(), align.bytes() as u32);
            }
            if regular.contains(ArgAttribute::ByVal) {
                llvm::LLVMRustAddByValAttr(llfn, idx.as_uint(), ty.unwrap());
            }
            regular.for_each_kind(|attr| attr.apply_llfn(idx, llfn));
        }
    }

    fn apply_callsite(&self, idx: AttributePlace, callsite: &Value, ty: Option<&Type>) {
        let mut regular = self.regular;
        unsafe {
            let deref = self.pointee_size.bytes();
            if deref != 0 {
                if regular.contains(ArgAttribute::NonNull) {
                    llvm::LLVMRustAddDereferenceableCallSiteAttr(callsite, idx.as_uint(), deref);
                } else {
                    llvm::LLVMRustAddDereferenceableOrNullCallSiteAttr(
                        callsite,
                        idx.as_uint(),
                        deref,
                    );
                }
                regular -= ArgAttribute::NonNull;
            }
            if let Some(align) = self.pointee_align {
                llvm::LLVMRustAddAlignmentCallSiteAttr(
                    callsite,
                    idx.as_uint(),
                    align.bytes() as u32,
                );
            }
            if regular.contains(ArgAttribute::ByVal) {
                llvm::LLVMRustAddByValCallSiteAttr(callsite, idx.as_uint(), ty.unwrap());
            }
            regular.for_each_kind(|attr| attr.apply_callsite(idx, callsite));
        }
    }
}

pub trait LlvmType {
    fn llvm_type(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type;
}

impl LlvmType for Reg {
    fn llvm_type(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type {
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
    fn llvm_type(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type {
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
            .flat_map(|option_kind| {
                option_kind.map(|kind| Reg { kind, size: self.prefix_chunk }.llvm_type(cx))
            })
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

impl ArgAbiExt<'ll, 'tcx> for ArgAbi<'tcx, Ty<'tcx>> {
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
        } else if let PassMode::Cast(cast) = self.mode {
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
        bx: &mut Builder<'a, 'll, 'tcx>,
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
            PassMode::Indirect(_, Some(_)) => {
                OperandValue::Ref(next(), Some(next()), self.layout.align.abi).store(bx, dst);
            }
            PassMode::Direct(_) | PassMode::Indirect(_, None) | PassMode::Cast(_) => {
                let next_arg = next();
                self.store(bx, next_arg, dst);
            }
        }
    }
}

impl ArgAbiMethods<'tcx> for Builder<'a, 'll, 'tcx> {
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

pub trait FnAbiLlvmExt<'tcx> {
    fn llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn ptr_to_llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn llvm_cconv(&self) -> llvm::CallConv;
    fn apply_attrs_llfn(&self, cx: &CodegenCx<'ll, 'tcx>, llfn: &'ll Value);
    fn apply_attrs_callsite(&self, bx: &mut Builder<'a, 'll, 'tcx>, callsite: &'ll Value);
}

impl<'tcx> FnAbiLlvmExt<'tcx> for FnAbi<'tcx, Ty<'tcx>> {
    fn llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type {
        let args_capacity: usize = self.args.iter().map(|arg|
            if arg.pad.is_some() { 1 } else { 0 } +
            if let PassMode::Pair(_, _) = arg.mode { 2 } else { 1 }
        ).sum();
        let mut llargument_tys = Vec::with_capacity(
            if let PassMode::Indirect(..) = self.ret.mode { 1 } else { 0 } + args_capacity,
        );

        let llreturn_ty = match self.ret.mode {
            PassMode::Ignore => cx.type_void(),
            PassMode::Direct(_) | PassMode::Pair(..) => self.ret.layout.immediate_llvm_type(cx),
            PassMode::Cast(cast) => cast.llvm_type(cx),
            PassMode::Indirect(..) => {
                llargument_tys.push(cx.type_ptr_to(self.ret.memory_ty(cx)));
                cx.type_void()
            }
        };

        for arg in &self.args {
            // add padding
            if let Some(ty) = arg.pad {
                llargument_tys.push(ty.llvm_type(cx));
            }

            let llarg_ty = match arg.mode {
                PassMode::Ignore => continue,
                PassMode::Direct(_) => arg.layout.immediate_llvm_type(cx),
                PassMode::Pair(..) => {
                    llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 0, true));
                    llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 1, true));
                    continue;
                }
                PassMode::Indirect(_, Some(_)) => {
                    let ptr_ty = cx.tcx.mk_mut_ptr(arg.layout.ty);
                    let ptr_layout = cx.layout_of(ptr_ty);
                    llargument_tys.push(ptr_layout.scalar_pair_element_llvm_type(cx, 0, true));
                    llargument_tys.push(ptr_layout.scalar_pair_element_llvm_type(cx, 1, true));
                    continue;
                }
                PassMode::Cast(cast) => cast.llvm_type(cx),
                PassMode::Indirect(_, None) => cx.type_ptr_to(arg.memory_ty(cx)),
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
                cx.data_layout().instruction_address_space as c_uint,
            )
        }
    }

    fn llvm_cconv(&self) -> llvm::CallConv {
        match self.conv {
            Conv::C | Conv::Rust => llvm::CCallConv,
            Conv::AmdGpuKernel => llvm::AmdGpuKernel,
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
        // FIXME(eddyb) can this also be applied to callsites?
        if self.ret.layout.abi.is_uninhabited() {
            llvm::Attribute::NoReturn.apply_llfn(llvm::AttributePlace::Function, llfn);
        }

        let mut i = 0;
        let mut apply = |attrs: &ArgAttributes, ty: Option<&Type>| {
            attrs.apply_llfn(llvm::AttributePlace::Argument(i), llfn, ty);
            i += 1;
        };
        match self.ret.mode {
            PassMode::Direct(ref attrs) => {
                attrs.apply_llfn(llvm::AttributePlace::ReturnValue, llfn, None);
            }
            PassMode::Indirect(ref attrs, _) => apply(attrs, Some(self.ret.layout.llvm_type(cx))),
            _ => {}
        }
        for arg in &self.args {
            if arg.pad.is_some() {
                apply(&ArgAttributes::new(), None);
            }
            match arg.mode {
                PassMode::Ignore => {}
                PassMode::Direct(ref attrs) | PassMode::Indirect(ref attrs, None) => {
                    apply(attrs, Some(arg.layout.llvm_type(cx)))
                }
                PassMode::Indirect(ref attrs, Some(ref extra_attrs)) => {
                    apply(attrs, None);
                    apply(extra_attrs, None);
                }
                PassMode::Pair(ref a, ref b) => {
                    apply(a, None);
                    apply(b, None);
                }
                PassMode::Cast(_) => apply(&ArgAttributes::new(), None),
            }
        }
    }

    fn apply_attrs_callsite(&self, bx: &mut Builder<'a, 'll, 'tcx>, callsite: &'ll Value) {
        let mut i = 0;
        let mut apply = |attrs: &ArgAttributes, ty: Option<&Type>| {
            attrs.apply_callsite(llvm::AttributePlace::Argument(i), callsite, ty);
            i += 1;
        };
        match self.ret.mode {
            PassMode::Direct(ref attrs) => {
                attrs.apply_callsite(llvm::AttributePlace::ReturnValue, callsite, None);
            }
            PassMode::Indirect(ref attrs, _) => apply(attrs, Some(self.ret.layout.llvm_type(bx))),
            _ => {}
        }
        if let layout::Abi::Scalar(ref scalar) = self.ret.layout.abi {
            // If the value is a boolean, the range is 0..2 and that ultimately
            // become 0..0 when the type becomes i1, which would be rejected
            // by the LLVM verifier.
            if let layout::Int(..) = scalar.value {
                if !scalar.is_bool() {
                    let range = scalar.valid_range_exclusive(bx);
                    if range.start != range.end {
                        bx.range_metadata(callsite, range);
                    }
                }
            }
        }
        for arg in &self.args {
            if arg.pad.is_some() {
                apply(&ArgAttributes::new(), None);
            }
            match arg.mode {
                PassMode::Ignore => {}
                PassMode::Direct(ref attrs) | PassMode::Indirect(ref attrs, None) => {
                    apply(attrs, Some(arg.layout.llvm_type(bx)))
                }
                PassMode::Indirect(ref attrs, Some(ref extra_attrs)) => {
                    apply(attrs, None);
                    apply(extra_attrs, None);
                }
                PassMode::Pair(ref a, ref b) => {
                    apply(a, None);
                    apply(b, None);
                }
                PassMode::Cast(_) => apply(&ArgAttributes::new(), None),
            }
        }

        let cconv = self.llvm_cconv();
        if cconv != llvm::CCallConv {
            llvm::SetInstructionCallConv(callsite, cconv);
        }
    }
}

impl AbiBuilderMethods<'tcx> for Builder<'a, 'll, 'tcx> {
    fn apply_attrs_callsite(&mut self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>, callsite: Self::Value) {
        fn_abi.apply_attrs_callsite(self, callsite)
    }

    fn get_param(&self, index: usize) -> Self::Value {
        llvm::get_param(self.llfn(), index as c_uint)
    }
}
