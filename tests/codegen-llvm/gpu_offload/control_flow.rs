//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// This test verifies that the offload intrinsic is correctly lowered even when the caller
// contains control flow.

#![feature(abi_gpu_kernel)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![no_main]

// CHECK: define{{( dso_local)?}} void @main()
// CHECK-NOT: define
// CHECK: %EmptyDesc = alloca %struct.__tgt_bin_desc, align 8
// CHECK-NEXT: %.offload_baseptrs = alloca [1 x ptr], align 8
// CHECK-NEXT: %.offload_ptrs = alloca [1 x ptr], align 8
// CHECK-NEXT: %.offload_sizes = alloca [1 x i64], align 8
// CHECK-NEXT: %kernel_args = alloca %struct.__tgt_kernel_arguments, align 8
// CHECK: br label %bb3
// CHECK-NOT define
// CHECK: bb3
// CHECK: call void @__tgt_target_data_begin_mapper(ptr nonnull @anon.{{.*}}.1, i64 -1, i32 1, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr nonnull @.offload_maptypes.foo, ptr null, ptr null)
// CHECK: %10 = call i32 @__tgt_target_kernel(ptr nonnull @anon.{{.*}}.1, i64 -1, i32 256, i32 32, ptr nonnull @.foo.region_id, ptr nonnull %kernel_args)
// CHECK-NEXT: call void @__tgt_target_data_end_mapper(ptr nonnull @anon.{{.*}}.1, i64 -1, i32 1, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr nonnull @.offload_maptypes.foo, ptr null, ptr null)
#[unsafe(no_mangle)]
unsafe fn main() {
    let A = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    for i in 0..100 {
        core::intrinsics::offload::<_, _, ()>(
            foo,
            [256, 1, 1],
            [32, 1, 1],
            (A.as_ptr() as *const [f32; 6],),
        );
    }
}

unsafe extern "C" {
    pub fn foo(A: *const [f32; 6]) -> ();
}
