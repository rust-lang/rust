//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=0 -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// This test verifies that selecting an unavailable `device` in the `offload` macro panics.

#![feature(gpu_offload)]
#![no_main]

#[unsafe(no_mangle)]
fn main() {
    core::offload::offload! {
        kernel = kernel,
        device = 99,
        args = (),
    }
}

#[unsafe(no_mangle)]
fn kernel() {}

// CHECK-LABEL: define{{( dso_local)?}} void @main()
// CHECK: store i32 99, ptr %device, align 4
// CHECK-NEXT: %{{[0-9_]+}} = call i32 @omp_get_num_devices()
// CHECK-NEXT: %{{[0-9_]+}} = load i32, ptr %device, align 4
// CHECK-NEXT: %{{[0-9_]+}} = icmp slt i32 %{{[0-9_]+}}, %{{[0-9_]+}}
// CHECK-NEXT: br i1 %{{[0-9_]+}}, label %bb{{[0-9]+}}, label %bb{{[0-9]+}}
// CHECK: call void @{{.*}}panic_fmt
// CHECK: unreachable
// CHECK: call i32 @__tgt_target_kernel
