//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=1 -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// This test verifies that the offload intrinsic is properly handling scalar args, passing them to
// the kernel as i64

#![feature(abi_gpu_kernel)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![no_main]

// CHECK: define{{( dso_local)?}} void @main()
// CHECK-NOT: define
// CHECK: %addr = alloca i64, align 8
// CHECK: store float 4.200000e+01, ptr [[TMP:%[^,]+]], align 4
// CHECK: [[VAL:%[0-9]+]] = load i32, ptr [[TMP]], align 4
// CHECK: [[VAL_I64:%[0-9]+]] = zext i32 [[VAL]] to i64
// CHECK: store i64 [[VAL_I64]], ptr %addr, align 8
// CHECK: [[REG_GEP1:%[^,]+]] = getelementptr inbounds nuw i8, ptr %.offload_baseptrs, i64 8
// CHECK-NEXT: store i64 [[VAL_I64]], ptr [[REG_GEP1]], align 8
// CHECK-NEXT: [[REG_GEP2:%[^,]+]] = getelementptr inbounds nuw i8, ptr %.offload_ptrs, i64 8
// CHECK-NEXT: store ptr %addr, ptr [[REG_GEP2]], align 8
// CHECK-NEXT: call void @__tgt_target_data_begin_mapper

#[unsafe(no_mangle)]
fn main() {
    let mut x = 0.0f32;
    let k = core::hint::black_box(42.0f32);

    core::intrinsics::offload::<_, _, ()>(foo, [1, 1, 1], [1, 1, 1], 0, (&mut x as *mut f32, k));
}

unsafe extern "C" {
    pub fn foo(x: *mut f32, k: f32);
}
