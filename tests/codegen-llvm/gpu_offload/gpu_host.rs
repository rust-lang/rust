//@ compile-flags: -Zoffload=Enable -Zunstable-options -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

// This test is verifying that we generate __tgt_target_data_*_mapper before and after a call to the
// kernel_1. Better documentation to what each global or variable means is available in the gpu
// offlaod code, or the LLVM offload documentation. This code does not launch any GPU kernels yet,
// and will be rewritten once a proper offload frontend has landed.
//
// We currently only handle memory transfer for specific calls to functions named `kernel_{num}`,
// when inside of a function called main. This, too, is a temporary workaround for not having a
// frontend.

#![no_main]

#[unsafe(no_mangle)]
fn main() {
    let mut x = [3.0; 256];
    kernel_1(&mut x);
    core::hint::black_box(&x);
}

// CHECK: %struct.__tgt_offload_entry = type { i64, i16, i16, i32, ptr, ptr, i64, i64, ptr }
// CHECK: %struct.__tgt_kernel_arguments = type { i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, [3 x i32], [3 x i32], i32 }
// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }
// CHECK: %struct.__tgt_bin_desc = type { i32, ptr, ptr, ptr }

// CHECK: @.offload_sizes.1 = private unnamed_addr constant [1 x i64] [i64 1024]
// CHECK: @.offload_maptypes.1 = private unnamed_addr constant [1 x i64] [i64 3]
// CHECK: @.kernel_1.region_id = weak unnamed_addr constant i8 0
// CHECK: @.offloading.entry_name.1 = internal unnamed_addr constant [9 x i8] c"kernel_1\00", section ".llvm.rodata.offloading", align 1
// CHECK: @.offloading.entry.kernel_1 = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr @.kernel_1.region_id, ptr @.offloading.entry_name.1, i64 0, i64 0, ptr null }, section ".omp_offloading_entries", align 1
// CHECK: @my_struct_global2 = external global %struct.__tgt_kernel_arguments
// CHECK: @0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
// CHECK: @1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8

// CHECK:  Function Attrs:
// CHECK-NEXT: define{{( dso_local)?}} void @main()
// CHECK-NEXT: start:
// CHECK-NEXT:   %0 = alloca [8 x i8], align 8
// CHECK-NEXT:   %x = alloca [1024 x i8], align 16
// CHECK-NEXT:   %EmptyDesc = alloca %struct.__tgt_bin_desc, align 8
// CHECK-NEXT:   %.offload_baseptrs = alloca [1 x ptr], align 8
// CHECK-NEXT:   %.offload_ptrs = alloca [1 x ptr], align 8
// CHECK-NEXT:   %.offload_sizes = alloca [1 x i64], align 8
// CHECK-NEXT:   %x.addr = alloca ptr, align 8
// CHECK-NEXT:   store ptr %x, ptr %x.addr, align 8
// CHECK-NEXT:   %1 = load ptr, ptr %x.addr, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds float, ptr %1, i32 0
// CHECK:        call void @llvm.memset.p0.i64(ptr align 8 %EmptyDesc, i8 0, i64 32, i1 false)
// CHECK-NEXT:   call void @__tgt_register_lib(ptr %EmptyDesc)
// CHECK-NEXT:   call void @__tgt_init_all_rtls()
// CHECK-NEXT:   %3 = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK-NEXT:   store ptr %1, ptr %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK-NEXT:   store ptr %2, ptr %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds [1 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK-NEXT:   store i64 1024, ptr %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK-NEXT:   %7 = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK-NEXT:   %8 = getelementptr inbounds [1 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK-NEXT:   call void @__tgt_target_data_begin_mapper(ptr @1, i64 -1, i32 1, ptr %6, ptr %7, ptr %8, ptr @.offload_maptypes.1, ptr null, ptr null)
// CHECK-NEXT:   call void @kernel_1(ptr noalias noundef nonnull align 4 dereferenceable(1024) %x)
// CHECK-NEXT:   %9 = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK-NEXT:   %10 = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK-NEXT:   %11 = getelementptr inbounds [1 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK-NEXT:   call void @__tgt_target_data_end_mapper(ptr @1, i64 -1, i32 1, ptr %9, ptr %10, ptr %11, ptr @.offload_maptypes.1, ptr null, ptr null)
// CHECK-NEXT:   call void @__tgt_unregister_lib(ptr %EmptyDesc)
// CHECK:        store ptr %x, ptr %0, align 8
// CHECK-NEXT:   call void asm sideeffect "", "r,~{memory}"(ptr nonnull %0)
// CHECK:        ret void
// CHECK-NEXT: }

#[unsafe(no_mangle)]
#[inline(never)]
pub fn kernel_1(x: &mut [f32; 256]) {
    for i in 0..256 {
        x[i] = 21.0;
    }
}
