use gccjit::{ToRValue, Type};
use rustc_codegen_ssa::traits::{AbiBuilderMethods, BaseTypeMethods};
use rustc_middle::bug;
use rustc_middle::ty::Ty;
use rustc_target::abi::call::{CastTarget, FnAbi, PassMode, Reg, RegKind};

use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::intrinsic::ArgAbiExt;
use crate::type_of::LayoutGccExt;

impl<'a, 'gcc, 'tcx> AbiBuilderMethods<'tcx> for Builder<'a, 'gcc, 'tcx> {
    fn apply_attrs_callsite(&mut self, _fn_abi: &FnAbi<'tcx, Ty<'tcx>>, _callsite: Self::Value) {
        // TODO
        //fn_abi.apply_attrs_callsite(self, callsite)
    }

    fn get_param(&self, index: usize) -> Self::Value {
        self.cx.current_func.borrow().expect("current func")
            .get_param(index as i32)
            .to_rvalue()
    }
}

impl GccType for CastTarget {
    fn gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, '_>) -> Type<'gcc> {
        let rest_gcc_unit = self.rest.unit.gcc_type(cx);
        let (rest_count, rem_bytes) =
            if self.rest.unit.size.bytes() == 0 {
                (0, 0)
            }
            else {
                (self.rest.total.bytes() / self.rest.unit.size.bytes(), self.rest.total.bytes() % self.rest.unit.size.bytes())
            };

        if self.prefix.iter().all(|x| x.is_none()) {
            // Simplify to a single unit when there is no prefix and size <= unit size
            if self.rest.total <= self.rest.unit.size {
                return rest_gcc_unit;
            }

            // Simplify to array when all chunks are the same size and type
            if rem_bytes == 0 {
                return cx.type_array(rest_gcc_unit, rest_count);
            }
        }

        // Create list of fields in the main structure
        let mut args: Vec<_> = self
            .prefix
            .iter()
            .flat_map(|option_kind| {
                option_kind.map(|kind| Reg { kind, size: self.prefix_chunk_size }.gcc_type(cx))
            })
            .chain((0..rest_count).map(|_| rest_gcc_unit))
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

pub trait GccType {
    fn gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, '_>) -> Type<'gcc>;
}

impl GccType for Reg {
    fn gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, '_>) -> Type<'gcc> {
        match self.kind {
            RegKind::Integer => cx.type_ix(self.size.bits()),
            RegKind::Float => {
                match self.size.bits() {
                    32 => cx.type_f32(),
                    64 => cx.type_f64(),
                    _ => bug!("unsupported float: {:?}", self),
                }
            },
            RegKind::Vector => unimplemented!(), //cx.type_vector(cx.type_i8(), self.size.bytes()),
        }
    }
}

pub trait FnAbiGccExt<'gcc, 'tcx> {
    // TODO: return a function pointer type instead?
    fn gcc_type(&self, cx: &CodegenCx<'gcc, 'tcx>) -> (Type<'gcc>, Vec<Type<'gcc>>, bool);
    fn ptr_to_gcc_type(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc>;
    /*fn llvm_cconv(&self) -> llvm::CallConv;
    fn apply_attrs_llfn(&self, cx: &CodegenCx<'ll, 'tcx>, llfn: &'ll Value);
    fn apply_attrs_callsite(&self, bx: &mut Builder<'a, 'll, 'tcx>, callsite: &'ll Value);*/
}

impl<'gcc, 'tcx> FnAbiGccExt<'gcc, 'tcx> for FnAbi<'tcx, Ty<'tcx>> {
    fn gcc_type(&self, cx: &CodegenCx<'gcc, 'tcx>) -> (Type<'gcc>, Vec<Type<'gcc>>, bool) {
        let args_capacity: usize = self.args.iter().map(|arg|
            if arg.pad.is_some() {
                1
            }
            else {
                0
            } +
            if let PassMode::Pair(_, _) = arg.mode {
                2
            } else {
                1
            }
        ).sum();
        let mut argument_tys = Vec::with_capacity(
            if let PassMode::Indirect { .. } = self.ret.mode {
                1
            }
            else {
                0
            } + args_capacity,
        );

        let return_ty =
            match self.ret.mode {
                PassMode::Ignore => cx.type_void(),
                PassMode::Direct(_) | PassMode::Pair(..) => self.ret.layout.immediate_gcc_type(cx),
                PassMode::Cast(cast) => cast.gcc_type(cx),
                PassMode::Indirect { .. } => {
                    argument_tys.push(cx.type_ptr_to(self.ret.memory_ty(cx)));
                    cx.type_void()
                }
            };

        for arg in &self.args {
            // add padding
            if let Some(ty) = arg.pad {
                argument_tys.push(ty.gcc_type(cx));
            }

            let arg_ty = match arg.mode {
                PassMode::Ignore => continue,
                PassMode::Direct(_) => arg.layout.immediate_gcc_type(cx),
                PassMode::Pair(..) => {
                    argument_tys.push(arg.layout.scalar_pair_element_gcc_type(cx, 0, true));
                    argument_tys.push(arg.layout.scalar_pair_element_gcc_type(cx, 1, true));
                    continue;
                }
                PassMode::Indirect { extra_attrs: Some(_), .. } => {
                    /*let ptr_ty = cx.tcx.mk_mut_ptr(arg.layout.ty);
                    let ptr_layout = cx.layout_of(ptr_ty);
                    argument_tys.push(ptr_layout.scalar_pair_element_gcc_type(cx, 0, true));
                    argument_tys.push(ptr_layout.scalar_pair_element_gcc_type(cx, 1, true));*/
                    unimplemented!();
                    //continue;
                }
                PassMode::Cast(cast) => cast.gcc_type(cx),
                PassMode::Indirect { extra_attrs: None, .. } => cx.type_ptr_to(arg.memory_ty(cx)),
            };
            argument_tys.push(arg_ty);
        }

        (return_ty, argument_tys, self.c_variadic)
    }

    fn ptr_to_gcc_type(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc> {
        let (return_type, params, variadic) = self.gcc_type(cx);
        let pointer_type = cx.context.new_function_pointer_type(None, return_type, &params, variadic);
        pointer_type
    }

    /*fn llvm_cconv(&self) -> llvm::CallConv {
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

        // FIXME(eddyb, wesleywiser): apply this to callsites as well?
        if !self.can_unwind {
            llvm::Attribute::NoUnwind.apply_llfn(llvm::AttributePlace::Function, llfn);
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
            PassMode::Indirect(ref attrs, _) => apply(attrs, Some(self.ret.layout.gcc_type(cx))),
            _ => {}
        }
        for arg in &self.args {
            if arg.pad.is_some() {
                apply(&ArgAttributes::new(), None);
            }
            match arg.mode {
                PassMode::Ignore => {}
                PassMode::Direct(ref attrs) | PassMode::Indirect(ref attrs, None) => {
                    apply(attrs, Some(arg.layout.gcc_type(cx)))
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
        // FIXME(wesleywiser, eddyb): We should apply `nounwind` and `noreturn` as appropriate to this callsite.

        let mut i = 0;
        let mut apply = |attrs: &ArgAttributes, ty: Option<&Type>| {
            attrs.apply_callsite(llvm::AttributePlace::Argument(i), callsite, ty);
            i += 1;
        };
        match self.ret.mode {
            PassMode::Direct(ref attrs) => {
                attrs.apply_callsite(llvm::AttributePlace::ReturnValue, callsite, None);
            }
            PassMode::Indirect(ref attrs, _) => apply(attrs, Some(self.ret.layout.gcc_type(bx))),
            _ => {}
        }
        if let abi::Abi::Scalar(ref scalar) = self.ret.layout.abi {
            // If the value is a boolean, the range is 0..2 and that ultimately
            // become 0..0 when the type becomes i1, which would be rejected
            // by the LLVM verifier.
            if let Int(..) = scalar.value {
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
                    apply(attrs, Some(arg.layout.gcc_type(bx)))
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
    }*/
}
