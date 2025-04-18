#[cfg(feature = "master")]
use gccjit::FnAttribute;
use gccjit::{ToLValue, ToRValue, Type};
use rustc_abi::{Reg, RegKind};
use rustc_codegen_ssa::traits::{AbiBuilderMethods, BaseTypeCodegenMethods};
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::bug;
use rustc_middle::ty::Ty;
use rustc_middle::ty::layout::LayoutOf;
#[cfg(feature = "master")]
use rustc_session::config;
#[cfg(feature = "master")]
use rustc_target::callconv::Conv;
use rustc_target::callconv::{ArgAttributes, CastTarget, FnAbi, PassMode};

use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::intrinsic::ArgAbiExt;
use crate::type_of::LayoutGccExt;

impl AbiBuilderMethods for Builder<'_, '_, '_> {
    fn get_param(&mut self, index: usize) -> Self::Value {
        let func = self.current_func();
        let param = func.get_param(index as i32);
        let on_stack = if let Some(on_stack_param_indices) =
            self.on_stack_function_params.borrow().get(&func)
        {
            on_stack_param_indices.contains(&index)
        } else {
            false
        };
        if on_stack { param.to_lvalue().get_address(None) } else { param.to_rvalue() }
    }
}

impl GccType for CastTarget {
    fn gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, '_>) -> Type<'gcc> {
        let rest_gcc_unit = self.rest.unit.gcc_type(cx);
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
            .flat_map(|option_reg| option_reg.map(|reg| reg.gcc_type(cx)))
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
            RegKind::Float => match self.size.bits() {
                32 => cx.type_f32(),
                64 => cx.type_f64(),
                _ => bug!("unsupported float: {:?}", self),
            },
            RegKind::Vector => unimplemented!(), //cx.type_vector(cx.type_i8(), self.size.bytes()),
        }
    }
}

pub struct FnAbiGcc<'gcc> {
    pub return_type: Type<'gcc>,
    pub arguments_type: Vec<Type<'gcc>>,
    pub is_c_variadic: bool,
    pub on_stack_param_indices: FxHashSet<usize>,
    #[cfg(feature = "master")]
    pub fn_attributes: Vec<FnAttribute<'gcc>>,
}

pub trait FnAbiGccExt<'gcc, 'tcx> {
    // TODO(antoyo): return a function pointer type instead?
    fn gcc_type(&self, cx: &CodegenCx<'gcc, 'tcx>) -> FnAbiGcc<'gcc>;
    fn ptr_to_gcc_type(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc>;
    #[cfg(feature = "master")]
    fn gcc_cconv(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Option<FnAttribute<'gcc>>;
}

impl<'gcc, 'tcx> FnAbiGccExt<'gcc, 'tcx> for FnAbi<'tcx, Ty<'tcx>> {
    fn gcc_type(&self, cx: &CodegenCx<'gcc, 'tcx>) -> FnAbiGcc<'gcc> {
        let mut on_stack_param_indices = FxHashSet::default();

        // This capacity calculation is approximate.
        let mut argument_tys = Vec::with_capacity(
            self.args.len() + if let PassMode::Indirect { .. } = self.ret.mode { 1 } else { 0 },
        );

        let return_type = match self.ret.mode {
            PassMode::Ignore => cx.type_void(),
            PassMode::Direct(_) | PassMode::Pair(..) => self.ret.layout.immediate_gcc_type(cx),
            PassMode::Cast { ref cast, .. } => cast.gcc_type(cx),
            PassMode::Indirect { .. } => {
                argument_tys.push(cx.type_ptr_to(self.ret.memory_ty(cx)));
                cx.type_void()
            }
        };
        #[cfg(feature = "master")]
        let mut non_null_args = Vec::new();

        #[cfg(feature = "master")]
        let mut apply_attrs = |mut ty: Type<'gcc>, attrs: &ArgAttributes, arg_index: usize| {
            if cx.sess().opts.optimize == config::OptLevel::No {
                return ty;
            }
            if attrs.regular.contains(rustc_target::callconv::ArgAttribute::NoAlias) {
                ty = ty.make_restrict()
            }
            if attrs.regular.contains(rustc_target::callconv::ArgAttribute::NonNull) {
                non_null_args.push(arg_index as i32 + 1);
            }
            ty
        };
        #[cfg(not(feature = "master"))]
        let apply_attrs = |ty: Type<'gcc>, _attrs: &ArgAttributes, _arg_index: usize| ty;

        for arg in self.args.iter() {
            let arg_ty = match arg.mode {
                PassMode::Ignore => continue,
                PassMode::Pair(a, b) => {
                    let arg_pos = argument_tys.len();
                    argument_tys.push(apply_attrs(
                        arg.layout.scalar_pair_element_gcc_type(cx, 0),
                        &a,
                        arg_pos,
                    ));
                    argument_tys.push(apply_attrs(
                        arg.layout.scalar_pair_element_gcc_type(cx, 1),
                        &b,
                        arg_pos + 1,
                    ));
                    continue;
                }
                PassMode::Cast { ref cast, pad_i32 } => {
                    // add padding
                    if pad_i32 {
                        argument_tys.push(Reg::i32().gcc_type(cx));
                    }
                    let ty = cast.gcc_type(cx);
                    apply_attrs(ty, &cast.attrs, argument_tys.len())
                }
                PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: true } => {
                    // This is a "byval" argument, so we don't apply the `restrict` attribute on it.
                    on_stack_param_indices.insert(argument_tys.len());
                    arg.memory_ty(cx)
                }
                PassMode::Direct(attrs) => {
                    apply_attrs(arg.layout.immediate_gcc_type(cx), &attrs, argument_tys.len())
                }
                PassMode::Indirect { attrs, meta_attrs: None, on_stack: false } => {
                    apply_attrs(cx.type_ptr_to(arg.memory_ty(cx)), &attrs, argument_tys.len())
                }
                PassMode::Indirect { attrs, meta_attrs: Some(meta_attrs), on_stack } => {
                    assert!(!on_stack);
                    // Construct the type of a (wide) pointer to `ty`, and pass its two fields.
                    // Any two ABI-compatible unsized types have the same metadata type and
                    // moreover the same metadata value leads to the same dynamic size and
                    // alignment, so this respects ABI compatibility.
                    let ptr_ty = Ty::new_mut_ptr(cx.tcx, arg.layout.ty);
                    let ptr_layout = cx.layout_of(ptr_ty);
                    let typ1 = ptr_layout.scalar_pair_element_gcc_type(cx, 0);
                    let typ2 = ptr_layout.scalar_pair_element_gcc_type(cx, 1);
                    argument_tys.push(apply_attrs(typ1, &attrs, argument_tys.len()));
                    argument_tys.push(apply_attrs(typ2, &meta_attrs, argument_tys.len()));
                    continue;
                }
            };
            argument_tys.push(arg_ty);
        }

        #[cfg(feature = "master")]
        let fn_attrs = if non_null_args.is_empty() {
            Vec::new()
        } else {
            vec![FnAttribute::NonNull(non_null_args)]
        };

        FnAbiGcc {
            return_type,
            arguments_type: argument_tys,
            is_c_variadic: self.c_variadic,
            on_stack_param_indices,
            #[cfg(feature = "master")]
            fn_attributes: fn_attrs,
        }
    }

    fn ptr_to_gcc_type(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc> {
        // FIXME(antoyo): Should we do something with `FnAbiGcc::fn_attributes`?
        let FnAbiGcc { return_type, arguments_type, is_c_variadic, on_stack_param_indices, .. } =
            self.gcc_type(cx);
        let pointer_type =
            cx.context.new_function_pointer_type(None, return_type, &arguments_type, is_c_variadic);
        cx.on_stack_params.borrow_mut().insert(
            pointer_type.dyncast_function_ptr_type().expect("function ptr type"),
            on_stack_param_indices,
        );
        pointer_type
    }

    #[cfg(feature = "master")]
    fn gcc_cconv(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Option<FnAttribute<'gcc>> {
        conv_to_fn_attribute(self.conv, &cx.tcx.sess.target.arch)
    }
}

#[cfg(feature = "master")]
pub fn conv_to_fn_attribute<'gcc>(conv: Conv, arch: &str) -> Option<FnAttribute<'gcc>> {
    // TODO: handle the calling conventions returning None.
    let attribute = match conv {
        Conv::C
        | Conv::Rust
        | Conv::CCmseNonSecureCall
        | Conv::CCmseNonSecureEntry
        | Conv::RiscvInterrupt { .. } => return None,
        Conv::Cold => return None,
        Conv::PreserveMost => return None,
        Conv::PreserveAll => return None,
        Conv::GpuKernel => {
            // TODO(antoyo): remove clippy allow attribute when this is implemented.
            #[allow(clippy::if_same_then_else)]
            if arch == "amdgpu" {
                return None;
            } else if arch == "nvptx64" {
                return None;
            } else {
                panic!("Architecture {} does not support GpuKernel calling convention", arch);
            }
        }
        Conv::AvrInterrupt => return None,
        Conv::AvrNonBlockingInterrupt => return None,
        Conv::ArmAapcs => return None,
        Conv::Msp430Intr => return None,
        Conv::X86Fastcall => return None,
        Conv::X86Intr => return None,
        Conv::X86Stdcall => return None,
        Conv::X86ThisCall => return None,
        Conv::X86VectorCall => return None,
        Conv::X86_64SysV => FnAttribute::SysvAbi,
        Conv::X86_64Win64 => FnAttribute::MsAbi,
    };
    Some(attribute)
}
