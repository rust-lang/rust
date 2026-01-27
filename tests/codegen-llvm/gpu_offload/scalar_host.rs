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
// CHECK: store double 4.200000e+01, ptr %0, align 8
// CHECK: %_0.i = load double, ptr %0, align 8
// CHECK: store double %_0.i, ptr %addr, align 8
// CHECK: %1 = getelementptr inbounds nuw i8, ptr %.offload_baseptrs, i64 8
// CHECK-NEXT: store double %_0.i, ptr %1, align 8
// CHECK-NEXT: %2 = getelementptr inbounds nuw i8, ptr %.offload_ptrs, i64 8
// CHECK-NEXT: store ptr %addr, ptr %2, align 8
// CHECK-NEXT: %3 = getelementptr inbounds nuw i8, ptr %.offload_sizes, i64 8
// CHECK-NEXT: store i64 4, ptr %3, align 8
// CHECK-NEXT: call void @__tgt_target_data_begin_mapper

#[unsafe(no_mangle)]
fn main() {
    let mut x = 0.0;
    let k = core::hint::black_box(42.0);

    core::intrinsics::offload::<_, _, ()>(foo, [1, 1, 1], [1, 1, 1], (&mut x, k));
}

unsafe extern "C" {
    pub fn foo(x: *mut f32, k: f32);
}
