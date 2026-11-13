use rustc_abi::{
    AddressSpace, BackendRepr, CanonAbi, HasDataLayout, Reg, RegKind, TyAbiInterface, TyAndLayout,
};

use crate::callconv::{ArgAbi, ComputeCallConv, Uniform};

// For reference, see llvm-project/clang/lib/CodeGen/Targets/AMDGPU.cpp

/// If the given type is a (potentially nested) struct containing a single scalar, return
/// a `Uniform` for the contained, single element.
fn single_element_struct_to_reg<'a, Ty, C>(cx: &C, ty: TyAndLayout<'a, Ty>) -> Option<Uniform>
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    assert!(ty.is_aggregate(), "Only handles aggregate types");
    if ty.layout.fields.count() != 1 {
        return None;
    }
    let field = ty.field(cx, 0);
    match field.backend_repr {
        BackendRepr::SimdScalableVector { .. } => panic!("scalable vectors are unsupported"),
        BackendRepr::Scalar(_) => {
            // Check that the size is the same as the size for ty, so no extra padding
            let size = field.layout.size.bytes();
            if ty.layout.size.bytes() != size {
                return None;
            }

            // clang passes the inner type directly, we emulate it with fitting integer types
            match size {
                1 => Some(Uniform::new(Reg::i8(), field.layout.size)),
                2 => Some(Uniform::new(Reg::i16(), field.layout.size)),
                4 => Some(Uniform::new(Reg::i32(), field.layout.size)),
                8 => Some(Uniform::new(Reg::i64(), field.layout.size)),
                16 => Some(Uniform::new(Reg::i128(), field.layout.size)),
                s => panic!("Unhandled scalar of size {s} in amdgpu gpu-kernel ABI"),
            }
        }
        BackendRepr::SimdVector { element, .. } => {
            // Check that the size is the same as the size for ty, so no extra padding
            let size = field.layout.size.bytes();
            if ty.layout.size.bytes() != size {
                return None;
            }

            // clang passes the inner type directly, we emulate it with a vector of the same type.
            // The size is rounded up to the size of the complete type (including alignment).
            let reg = Reg {
                kind: RegKind::Vector { hint_vector_elem: element.primitive() },
                size: field.layout.size,
            };
            Some(Uniform::new(reg, field.layout.size))
        }
        BackendRepr::Memory { .. } => single_element_struct_to_reg(cx, field),
        BackendRepr::ScalarPair { .. } => None,
    }
}

pub(crate) struct AmdGpuCallConv;

impl<'a, Ty, C> ComputeCallConv<'a, Ty, C> for AmdGpuCallConv
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    fn new(_cx: &C) -> Self {
        Self
    }

    fn compute_ret_abi(&mut self, _cx: &C, _conv: CanonAbi, _ret: &mut ArgAbi<'a, Ty>) {
        // Kernels cannot return values, so do not handle return types
    }

    fn compute_arg_abi(
        &mut self,
        cx: &C,
        conv: CanonAbi,
        arg: &mut ArgAbi<'a, Ty>,
        _is_vararg: bool,
    ) {
        // Try to fill first registers with values and pass by_ref pointers for later indirect arguments

        if arg.is_ignore() {
            return;
        }
        if conv == CanonAbi::GpuKernel {
            if arg.layout.is_aggregate() {
                if let Some(uniform) = single_element_struct_to_reg(cx, arg.layout) {
                    // Single element structs are passed directly as the inner type
                    arg.cast_to(uniform);
                } else {
                    // All other aggregates are passed as by_ref pointer in the constant address space
                    arg.pass_amdgpu_kernel_arg(Some(AddressSpace::GPU_CONSTANT));
                }
            }
        } else {
            // FIXME: C ABI is not yet implemented
        }
    }
}
