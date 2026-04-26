//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=1 -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// This test verifies that offload is properly handling slices passing them properly to the device

#![feature(abi_gpu_kernel)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![no_main]

// CHECK: @anon.[[ID:.*]].0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1

// CHECK-DAG: @.offload_sizes.[[K:[^ ]*foo]] = private unnamed_addr constant [2 x i64] [i64 0, i64 8]
// CHECK-DAG: @.offload_maptypes.[[K]].begin = private unnamed_addr constant [2 x i64] [i64 1, i64 768]
// CHECK-DAG: @.offload_maptypes.[[K]].kernel = private unnamed_addr constant [2 x i64] [i64 32, i64 800]
// CHECK-DAG: @.offload_maptypes.[[K]].end = private unnamed_addr constant [2 x i64] [i64 2, i64 0]

// CHECK:       define{{( dso_local)?}} void @main()
// CHECK:       %.offload_sizes = alloca [2 x i64], align 8
// CHECK:  call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %.offload_sizes, ptr {{.*}} @.offload_sizes.foo, i64 16, i1 false)
// CHECK:       store i64 16, ptr %.offload_sizes, align 8
// CHECK:       call void @__tgt_target_data_begin_mapper(ptr nonnull @anon.[[ID]].1, i64 -1, i32 2, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr nonnull @.offload_maptypes.[[K]].begin, ptr null, ptr null)
// CHECK:       %11 = call i32 @__tgt_target_kernel(ptr nonnull @anon.[[ID]].1, i64 -1, i32 1, i32 1, ptr nonnull @.foo.region_id, ptr nonnull %kernel_args)
// CHECK-NEXT:  call void @__tgt_target_data_end_mapper(ptr nonnull @anon.[[ID]].1, i64 -1, i32 2, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr nonnull @.offload_maptypes.[[K]].end, ptr null, ptr null)

#[unsafe(no_mangle)]
fn main() {
    let mut x = [0.0, 0.0, 0.0, 0.0];
    core::intrinsics::offload::<_, _, ()>(foo, [1, 1, 1], [1, 1, 1], ((&mut x) as &mut [f64],));
}

unsafe extern "C" {
    pub fn foo(x: &mut [f32]);
}
