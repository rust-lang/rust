//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=1 -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// This test verifies that a `Region` kernel argument is mapped like a slice.
#![feature(abi_gpu_kernel)]
#![feature(core_intrinsics)]
#![feature(gpu_offload)]
#![feature(offload)]
#![feature(rustc_attrs)]
#![no_main]

extern crate core;

use core::offload::{PartitioningStrategy, Region};

struct Dummy;

unsafe impl PartitioningStrategy for Dummy {
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    fn index() -> usize {
        0
    }

    unsafe fn get<'a, T>(_ptr: *const T, _len: usize) -> Option<Self::View<'a, T>> {
        None
    }

    unsafe fn get_mut<'a, T>(_ptr: *mut T, _len: usize) -> Option<Self::ViewMut<'a, T>> {
        None
    }
}

// CHECK: @anon.[[ID:.*]].0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1

// CHECK-DAG: @.offload_sizes.[[K:[^ ]*foo]] = private unnamed_addr constant [2 x i64] [i64 0, i64 8]
// CHECK-DAG: @.offload_maptypes.[[K]].begin = private unnamed_addr constant [2 x i64] [i64 1, i64 768]
// CHECK-DAG: @.offload_maptypes.[[K]].kernel = private unnamed_addr constant [2 x i64] [i64 32, i64 800]
// CHECK-DAG: @.offload_maptypes.[[K]].end = private unnamed_addr constant [2 x i64] [i64 2, i64 0]

// CHECK:       define{{( dso_local)?}} void @main()
// CHECK:       %.offload_sizes = alloca [2 x i64], align 8
// CHECK:  call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %.offload_sizes, ptr {{.*}} @.offload_sizes.[[K]], i64 16, i1 false)
// CHECK:       store i64 16, ptr %.offload_sizes, align 8
// CHECK:       call void @__tgt_target_data_begin_mapper(ptr nonnull @anon.[[ID]].1, i64 -1, i32 2, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr nonnull @.offload_maptypes.[[K]].begin, ptr null, ptr null)
// CHECK:       call i32 @__tgt_target_kernel(ptr nonnull @anon.[[ID]].1, i64 -1, i32 1, i32 1, ptr nonnull @.[[K]].region_id, ptr nonnull %kernel_args)
// CHECK-NEXT:  call void @__tgt_target_data_end_mapper(ptr nonnull @anon.[[ID]].1, i64 -1, i32 2, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr nonnull @.offload_maptypes.[[K]].end, ptr null, ptr null)

#[unsafe(no_mangle)]
fn main() {
    let mut x = [0.0f32; 4];
    core::intrinsics::offload::<_, _, ()>(
        foo,
        [1, 1, 1],
        [1, 1, 1],
        0,
        (Region::<f32, Dummy>::new(&mut x as &mut [f32]),),
    );
}

fn foo(region: Region<'_, f32, Dummy>) {
    unreachable!();
}
