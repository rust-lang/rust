use gccjit::{ToLValue, ToRValue, Type};
use rustc_codegen_ssa::traits::{AbiBuilderMethods, BaseTypeMethods};
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::bug;
use rustc_middle::ty::Ty;
use rustc_target::abi::call::{CastTarget, FnAbi, PassMode, Reg, RegKind};

use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::intrinsic::ArgAbiExt;
use crate::type_of::LayoutGccExt;

impl<'a, 'gcc, 'tcx> AbiBuilderMethods<'tcx> for Builder<'a, 'gcc, 'tcx> {
    fn apply_attrs_callsite(&mut self, _fn_abi: &FnAbi<'tcx, Ty<'tcx>>, _callsite: Self::Value) {
        // TODO(antoyo)
    }

    fn get_param(&mut self, index: usize) -> Self::Value {
        let func = self.current_func();
        let param = func.get_param(index as i32);
        let on_stack =
            if let Some(on_stack_param_indices) = self.on_stack_function_params.borrow().get(&func) {
                on_stack_param_indices.contains(&index)
            }
            else {
                false
            };
        if on_stack {
            param.to_lvalue().get_address(None)
        }
        else {
            param.to_rvalue()
        }
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
            .flat_map(|option_reg| {
                option_reg.map(|reg| reg.gcc_type(cx))
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
    // TODO(antoyo): return a function pointer type instead?
    fn gcc_type(&self, cx: &CodegenCx<'gcc, 'tcx>) -> (Type<'gcc>, Vec<Type<'gcc>>, bool, FxHashSet<usize>);
    fn ptr_to_gcc_type(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc>;
}

impl<'gcc, 'tcx> FnAbiGccExt<'gcc, 'tcx> for FnAbi<'tcx, Ty<'tcx>> {
    fn gcc_type(&self, cx: &CodegenCx<'gcc, 'tcx>) -> (Type<'gcc>, Vec<Type<'gcc>>, bool, FxHashSet<usize>) {
        let mut on_stack_param_indices = FxHashSet::default();

        // This capacity calculation is approximate.
        let mut argument_tys = Vec::with_capacity(
            self.args.len() + if let PassMode::Indirect { .. } = self.ret.mode { 1 } else { 0 }
        );

        let return_ty =
            match self.ret.mode {
                PassMode::Ignore => cx.type_void(),
                PassMode::Direct(_) | PassMode::Pair(..) => self.ret.layout.immediate_gcc_type(cx),
                PassMode::Cast(ref cast, _) => cast.gcc_type(cx),
                PassMode::Indirect { .. } => {
                    argument_tys.push(cx.type_ptr_to(self.ret.memory_ty(cx)));
                    cx.type_void()
                }
            };

        for arg in self.args.iter() {
            let arg_ty = match arg.mode {
                PassMode::Ignore => continue,
                PassMode::Direct(_) => arg.layout.immediate_gcc_type(cx),
                PassMode::Pair(..) => {
                    argument_tys.push(arg.layout.scalar_pair_element_gcc_type(cx, 0, true));
                    argument_tys.push(arg.layout.scalar_pair_element_gcc_type(cx, 1, true));
                    continue;
                }
                PassMode::Indirect { extra_attrs: Some(_), .. } => {
                    unimplemented!();
                }
                PassMode::Cast(ref cast, pad_i32) => {
                    // add padding
                    if pad_i32 {
                        argument_tys.push(Reg::i32().gcc_type(cx));
                    }
                    cast.gcc_type(cx)
                }
                PassMode::Indirect { extra_attrs: None, on_stack: true, .. } => {
                    on_stack_param_indices.insert(argument_tys.len());
                    arg.memory_ty(cx)
                },
                PassMode::Indirect { extra_attrs: None, on_stack: false, .. } => cx.type_ptr_to(arg.memory_ty(cx)),
            };
            argument_tys.push(arg_ty);
        }

        (return_ty, argument_tys, self.c_variadic, on_stack_param_indices)
    }

    fn ptr_to_gcc_type(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc> {
        let (return_type, params, variadic, on_stack_param_indices) = self.gcc_type(cx);
        let pointer_type = cx.context.new_function_pointer_type(None, return_type, &params, variadic);
        cx.on_stack_params.borrow_mut().insert(pointer_type.dyncast_function_ptr_type().expect("function ptr type"), on_stack_param_indices);
        pointer_type
    }
}
