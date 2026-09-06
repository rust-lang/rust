use rustc_abi::{
    AddressSpace, BackendRepr, CanonAbi, HasDataLayout, Reg, TyAbiInterface, TyAndLayout,
};

use crate::callconv::{FnAbi, Uniform};

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
        BackendRepr::Scalar(_)
        | BackendRepr::SimdVector { .. }
        | BackendRepr::SimdScalableVector { .. } => {
            // Check that the size is the same as the size for ty, so no extra padding
            let size = field.layout.size.bytes();
            if ty.layout.size.bytes() != size {
                return None;
            }
            // clang passes the inner type directly, we emulate it with cast [n x i32] or smaller types
            match size {
                1 => Some(Uniform::new(Reg::i8(), field.layout.size)),
                2 => Some(Uniform::new(Reg::i16(), field.layout.size)),
                _ => Some(Uniform::new(Reg::i32(), field.layout.size)),
            }
        }
        BackendRepr::Memory { .. } => single_element_struct_to_reg(cx, field),
        BackendRepr::ScalarPair { .. } => None,
    }
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    // Kernels cannot return values, so do not handle return types

    // Try to fill first registers with values and pass by_ref pointers for later indirect arguments
    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        if fn_abi.conv == CanonAbi::GpuKernel {
            if arg.layout.is_aggregate() {
                if let Some(uniform) = single_element_struct_to_reg(cx, arg.layout) {
                    // Single element structs are passed directly as the inner type
                    arg.cast_to(uniform);
                } else {
                    // All other aggregates are passed as by_ref pointer in the constant address space
                    arg.pass_by_ref(Some(AddressSpace::GPU_CONSTANT));
                }
            }
        }
    }
}
