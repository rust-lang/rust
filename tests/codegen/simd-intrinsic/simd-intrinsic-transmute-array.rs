//
// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![allow(non_camel_case_types)]
#![feature(repr_simd, platform_intrinsics)]
#![feature(inline_const)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct S<const N: usize>([f32; N]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct T([f32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct U(f32, f32, f32, f32);

// CHECK-LABEL: @array_align(
#[no_mangle]
pub fn array_align() -> usize {
    // CHECK: ret [[USIZE:i[0-9]+]] [[ARRAY_ALIGN:[0-9]+]]
    const { std::mem::align_of::<f32>() }
}

// CHECK-LABEL: @vector_align(
#[no_mangle]
pub fn vector_align() -> usize {
    // CHECK: ret [[USIZE]] [[VECTOR_ALIGN:[0-9]+]]
    const { std::mem::align_of::<U>() }
}

// CHECK-LABEL: @build_array_s
#[no_mangle]
pub fn build_array_s(x: [f32; 4]) -> S<4> {
    // CHECK: call void @llvm.memcpy.{{.+}}({{.*}} align [[VECTOR_ALIGN]] {{.*}} align [[ARRAY_ALIGN]] {{.*}}, [[USIZE]] 16, i1 false)
    S::<4>(x)
}

// CHECK-LABEL: @build_array_transmute_s
#[no_mangle]
pub fn build_array_transmute_s(x: [f32; 4]) -> S<4> {
    // CHECK: %[[VAL:.+]] = load <4 x float>, {{ptr %x|.+>\* %.+}}, align [[ARRAY_ALIGN]]
    // CHECK: store <4 x float> %[[VAL:.+]], {{ptr %_0|.+>\* %.+}}, align [[VECTOR_ALIGN]]
    unsafe { std::mem::transmute(x) }
}

// CHECK-LABEL: @build_array_t
#[no_mangle]
pub fn build_array_t(x: [f32; 4]) -> T {
    // CHECK: call void @llvm.memcpy.{{.+}}({{.*}} align [[VECTOR_ALIGN]] {{.*}} align [[ARRAY_ALIGN]] {{.*}}, [[USIZE]] 16, i1 false)
    T(x)
}

// CHECK-LABEL: @build_array_transmute_t
#[no_mangle]
pub fn build_array_transmute_t(x: [f32; 4]) -> T {
    // CHECK: %[[VAL:.+]] = load <4 x float>, {{ptr %x|.+>\* %.+}}, align [[ARRAY_ALIGN]]
    // CHECK: store <4 x float> %[[VAL:.+]], {{ptr %_0|.+>\* %.+}}, align [[VECTOR_ALIGN]]
    unsafe { std::mem::transmute(x) }
}

// CHECK-LABEL: @build_array_u
#[no_mangle]
pub fn build_array_u(x: [f32; 4]) -> U {
    // CHECK: store float %a, {{.+}}, align [[VECTOR_ALIGN]]
    // CHECK: store float %b, {{.+}}, align [[ARRAY_ALIGN]]
    // CHECK: store float %c, {{.+}}, align
    // CHECK: store float %d, {{.+}}, align [[ARRAY_ALIGN]]
    let [a, b, c, d] = x;
    U(a, b, c, d)
}

// CHECK-LABEL: @build_array_transmute_u
#[no_mangle]
pub fn build_array_transmute_u(x: [f32; 4]) -> U {
    // CHECK: %[[VAL:.+]] = load <4 x float>, {{ptr %x|.+>\* %.+}}, align [[ARRAY_ALIGN]]
    // CHECK: store <4 x float> %[[VAL:.+]], {{ptr %_0|.+>\* %.+}}, align [[VECTOR_ALIGN]]
    unsafe { std::mem::transmute(x) }
}
