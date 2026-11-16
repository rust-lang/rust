//@ add-minicore
//@ compile-flags: --crate-type=rlib --target=amdgcn-amd-amdhsa -Ctarget-cpu=gfx900 -Copt-level=3
//@ needs-llvm-components: amdgpu
#![feature(no_core, abi_gpu_kernel, repr_simd)]
#![no_core]
#![allow(improper_gpu_kernel_arg)]

extern crate minicore;
use minicore::num::Complex;

// Tests from llvm-project/clang/test/CodeGenOpenCL/amdgpu-abi-struct-coerce.cl

#[repr(simd)]
pub struct I8X2([i8; 2]);

#[repr(simd)]
pub struct I16X2([i16; 2]);

#[repr(simd)]
pub struct I16X3([i16; 3]);

#[repr(simd)]
pub struct I16X4([i16; 4]);

#[repr(simd)]
pub struct I32X3([i32; 3]);

#[repr(simd)]
pub struct I32X4([i32; 4]);

#[repr(C)]
pub struct SingleElementStructArg<T> {
    i: T,
}

#[repr(C)]
pub struct NestedSingleElementStructArg {
    i: SingleElementStructArg<i32>,
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

// CHECK: define amdgpu_kernel void @kernel_single_element_struct_arg(i32 %0)
#[no_mangle]
pub extern "gpu-kernel" fn kernel_single_element_struct_arg(_: SingleElementStructArg<i32>) {}

// CHECK: define amdgpu_kernel void @kernel_nested_single_element_struct_arg(i32 %0)
#[no_mangle]
pub extern "gpu-kernel" fn kernel_nested_single_element_struct_arg(
    _: NestedSingleElementStructArg,
) {
}

// CHECK: define amdgpu_kernel void @kernel_struct_arg(ptr addrspace(4) noalias nofree noundef readnone byref([12 x i8]) align 4 captures(none) dereferenceable(12) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_struct_arg(_: StructArg) {}

// CHECK: define amdgpu_kernel void @kernel_struct_padding_arg(ptr addrspace(4) noalias nofree noundef readnone byref([16 x i8]) align 8 captures(none) dereferenceable(16) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_struct_padding_arg(_: StructPaddingArg) {}

// CHECK: define amdgpu_kernel void @kernel_struct_of_arrays_arg(ptr addrspace(4) noalias nofree noundef readnone byref([44 x i8]) align 4 captures(none) dereferenceable(44) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_struct_of_arrays_arg(_: StructOfArraysArg) {}

// CHECK: define amdgpu_kernel void @kernel_struct_of_structs_arg(ptr addrspace(4) noalias nofree noundef readnone byref([24 x i8]) align 4 captures(none) dereferenceable(24) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_struct_of_structs_arg(_: StructOfStructsArg) {}

// CHECK: define amdgpu_kernel void @test_kernel_union_arg(ptr addrspace(4) noalias nofree noundef readnone byref([4 x i8]) align 4 captures(none) dereferenceable(4) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn test_kernel_union_arg(_: U) {}

// CHECK: define amdgpu_kernel void @kernel_single_array_element_struct_arg(ptr addrspace(4) noalias nofree noundef readnone byref([16 x i8]) align 4 captures(none) dereferenceable(16) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_single_array_element_struct_arg(_: SingleArrayElementStructArg) {}

// CHECK: define amdgpu_kernel void @kernel_single_struct_element_struct_arg(ptr addrspace(4) noalias nofree noundef readnone byref([16 x i8]) align 8 captures(none) dereferenceable(16) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_single_struct_element_struct_arg(
    _: SingleStructElementStructArg,
) {
}

// CHECK: define amdgpu_kernel void @kernel_different_size_type_pair_arg(ptr addrspace(4) noalias nofree noundef readnone byref([16 x i8]) align 8 captures(none) dereferenceable(16) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_different_size_type_pair_arg(_: DifferentSizeTypePair) {}

// CHECK: define amdgpu_kernel void @kernel_complex(ptr addrspace(4) noalias nofree noundef readnone byref([8 x i8]) align 4 captures(none) dereferenceable(8) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_complex(_: Complex<f32>) {}

// CHECK: define amdgpu_kernel void @kernel_slice(ptr addrspace(4) noalias nofree noundef readnone byref([16 x i8]) align 8 captures(none) dereferenceable(16) {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_slice(_: &[u32]) {}

// CHECK: define amdgpu_kernel void @kernel_i64(i64 noundef {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_i64(_: i64) {}

// CHECK: define amdgpu_kernel void @kernel_i64_struct(i64 {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_i64_struct(_: SingleElementStructArg<i64>) {}

// CHECK: define amdgpu_kernel void @kernel_i128_struct(i128 {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_i128_struct(_: SingleElementStructArg<i128>) {}

// CHECK: define amdgpu_kernel void @kernel_i8x2_struct(<2 x i8> {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_i8x2_struct(_: SingleElementStructArg<I8X2>) {}

// CHECK: define amdgpu_kernel void @kernel_i16x2_struct(<2 x i16> {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_i16x2_struct(_: SingleElementStructArg<I16X2>) {}

// CHECK: define amdgpu_kernel void @kernel_i16x3_struct(<4 x i16> {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_i16x3_struct(_: SingleElementStructArg<I16X3>) {}

// CHECK: define amdgpu_kernel void @kernel_i16x4_struct(<4 x i16> {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_i16x4_struct(_: SingleElementStructArg<I16X4>) {}

// CHECK: define amdgpu_kernel void @kernel_i32x3_struct(<4 x i32> {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_i32x3_struct(_: SingleElementStructArg<I32X3>) {}

// CHECK: define amdgpu_kernel void @kernel_i32x4_struct(<4 x i32> {{%.+}})
#[no_mangle]
pub extern "gpu-kernel" fn kernel_i32x4_struct(_: SingleElementStructArg<I32X4>) {}
