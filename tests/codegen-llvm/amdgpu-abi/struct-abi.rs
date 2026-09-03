//@ add-minicore
//@ compile-flags: --crate-type=rlib --target=amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@ needs-llvm-components: amdgpu
#![feature(no_core, abi_gpu_kernel)]
#![no_core]
#![allow(improper_gpu_kernel_arg)]

extern crate minicore;
use minicore::num::Complex;

// Tests from llvm-project/clang/test/CodeGenOpenCL/amdgpu-abi-struct-coerce.cl

#[repr(C)]
pub struct SingleElementStructArg {
    i: i32,
}

#[repr(C)]
pub struct NestedSingleElementStructArg {
    i: SingleElementStructArg,
}

#[repr(C)]
pub struct StructArg {
    i1: i32,
    f: f32,
    i2: i32,
}

#[repr(C)]
pub struct StructPaddingArg {
    i1: i8,
    f: i64,
}

#[repr(C)]
pub struct StructOfArraysArg {
    i1: [i32; 2],
    f1: f32,
    i2: [i32; 4],
    f2: [f32; 3],
    i3: i32,
}

#[repr(C)]
pub struct StructOfStructsArg {
    i1: i32,
    f1: f32,
    s1: StructArg,
    i2: i32,
}

#[repr(C)]
pub union U {
    b1: i32,
    b2: f32,
}

#[repr(C)]
pub struct SingleArrayElementStructArg {
    i: [i32; 4],
}

#[repr(C)]
pub struct SingleStructElementStructArgInner {
    i: i32,
    b: i64,
}

#[repr(C)]
pub struct SingleStructElementStructArg {
    s: SingleStructElementStructArgInner,
}

#[repr(C)]
pub struct DifferentSizeTypePair {
    l: i64,
    i: i32,
}

// CHECK: define amdgpu_kernel void @kernel_single_element_struct_arg(ptr noalias nofree noundef readnone align 4 captures(none) dead_on_return dereferenceable(4) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_single_element_struct_arg(_: SingleElementStructArg) {}

// CHECK: define amdgpu_kernel void @kernel_nested_single_element_struct_arg(ptr noalias nofree noundef readnone align 4 captures(none) dead_on_return dereferenceable(4) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_nested_single_element_struct_arg(
    _: NestedSingleElementStructArg,
) {
}

// CHECK: define amdgpu_kernel void @kernel_struct_arg(ptr noalias nofree noundef readnone align 4 captures(none) dead_on_return dereferenceable(12) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_struct_arg(_: StructArg) {}

// CHECK: define amdgpu_kernel void @kernel_struct_padding_arg(i8 noundef {{%.+}}, i64 noundef {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_struct_padding_arg(_: StructPaddingArg) {}

// CHECK: define amdgpu_kernel void @kernel_struct_of_arrays_arg(ptr noalias nofree noundef readnone align 4 captures(none) dead_on_return dereferenceable(44) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_struct_of_arrays_arg(_: StructOfArraysArg) {}

// CHECK: define amdgpu_kernel void @kernel_struct_of_structs_arg(ptr noalias nofree noundef readnone align 4 captures(none) dead_on_return dereferenceable(24) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_struct_of_structs_arg(_: StructOfStructsArg) {}

// CHECK: define amdgpu_kernel void @test_kernel_union_arg(ptr noalias nofree noundef readnone align 4 captures(none) dead_on_return dereferenceable(4) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn test_kernel_union_arg(_: U) {}

// CHECK: define amdgpu_kernel void @kernel_single_array_element_struct_arg(ptr noalias nofree noundef readnone align 4 captures(none) dead_on_return dereferenceable(16) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_single_array_element_struct_arg(_: SingleArrayElementStructArg) {}

// CHECK: define amdgpu_kernel void @kernel_single_struct_element_struct_arg(i32 noundef {{%.+}}, i64 noundef {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_single_struct_element_struct_arg(
    _: SingleStructElementStructArg,
) {
}

// CHECK: define amdgpu_kernel void @kernel_different_size_type_pair_arg(i64 noundef {{%.+}}, i32 noundef {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_different_size_type_pair_arg(_: DifferentSizeTypePair) {}

// CHECK: define amdgpu_kernel void @kernel_complex(float noundef {{%.+}}, float noundef {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_complex(_: Complex<f32>) {}

// CHECK: define amdgpu_kernel void @kernel_slice(ptr noalias nofree noundef nonnull readonly align 4 captures(none) {{%.+}}, i64 noundef range(i64 0, 2305843009213693952) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_slice(_: &[u32]) {}
