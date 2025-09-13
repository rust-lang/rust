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
// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }
// CHECK: %struct.__tgt_bin_desc = type { i32, ptr, ptr, ptr }
// CHECK: %struct.__tgt_kernel_arguments = type { i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, [3 x i32], [3 x i32], i32 }

// CHECK: @.offload_sizes.1 = private unnamed_addr constant [1 x i64] [i64 1024]
// CHECK: @.offload_maptypes.1 = private unnamed_addr constant [1 x i64] [i64 35]
// CHECK: @.kernel_1.region_id = weak unnamed_addr constant i8 0
// CHECK: @.offloading.entry_name.1 = internal unnamed_addr constant [9 x i8] c"kernel_1\00", section ".llvm.rodata.offloading", align 1
// CHECK: @.offloading.entry.kernel_1 = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr @.kernel_1.region_id, ptr @.offloading.entry_name.1, i64 0, i64 0, ptr null }, section "llvm_offload_entries", align 8
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
// CHECK-NEXT:   %kernel_args = alloca %struct.__tgt_kernel_arguments, align 8
// CHECK:        call void @llvm.memset.p0.i64(ptr align 8 %EmptyDesc, i8 0, i64 32, i1 false)
// CHECK-NEXT:   %1 = getelementptr inbounds float, ptr %x, i32 0
// CHECK-NEXT:   call void @__tgt_register_lib(ptr %EmptyDesc)
// CHECK-NEXT:   call void @__tgt_init_all_rtls()
// CHECK-NEXT:   %2 = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK-NEXT:   store ptr %x, ptr %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK-NEXT:   store ptr %1, ptr %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds [1 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK-NEXT:   store i64 1024, ptr %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK-NEXT:   %6 = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK-NEXT:   %7 = getelementptr inbounds [1 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK-NEXT:   call void @__tgt_target_data_begin_mapper(ptr @1, i64 -1, i32 1, ptr %5, ptr %6, ptr %7, ptr @.offload_maptypes.1, ptr null, ptr null)
// CHECK-NEXT:   %8 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 0
// CHECK-NEXT:   store i32 3, ptr %8, align 4
// CHECK-NEXT:   %9 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 1
// CHECK-NEXT:   store i32 1, ptr %9, align 4
// CHECK-NEXT:   %10 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 2
// CHECK-NEXT:   store ptr %5, ptr %10, align 8
// CHECK-NEXT:   %11 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 3
// CHECK-NEXT:   store ptr %6, ptr %11, align 8
// CHECK-NEXT:   %12 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 4
// CHECK-NEXT:   store ptr %7, ptr %12, align 8
// CHECK-NEXT:   %13 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 5
// CHECK-NEXT:   store ptr @.offload_maptypes.1, ptr %13, align 8
// CHECK-NEXT:   %14 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 6
// CHECK-NEXT:   store ptr null, ptr %14, align 8
// CHECK-NEXT:   %15 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 7
// CHECK-NEXT:   store ptr null, ptr %15, align 8
// CHECK-NEXT:   %16 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// CHECK-NEXT:   store i64 0, ptr %16, align 8
// CHECK-NEXT:   %17 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 9
// CHECK-NEXT:   store i64 0, ptr %17, align 8
// CHECK-NEXT:   %18 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 10
// CHECK-NEXT:   store [3 x i32] [i32 2097152, i32 0, i32 0], ptr %18, align 4
// CHECK-NEXT:   %19 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 11
// CHECK-NEXT:   store [3 x i32] [i32 256, i32 0, i32 0], ptr %19, align 4
// CHECK-NEXT:   %20 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 12
// CHECK-NEXT:   store i32 0, ptr %20, align 4
// CHECK-NEXT:   %21 = call i32 @__tgt_target_kernel(ptr @1, i64 -1, i32 2097152, i32 256, ptr @.kernel_1.region_id, ptr %kernel_args)
// CHECK-NEXT:   %22 = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK-NEXT:   %23 = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK-NEXT:   %24 = getelementptr inbounds [1 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK-NEXT:   call void @__tgt_target_data_end_mapper(ptr @1, i64 -1, i32 1, ptr %22, ptr %23, ptr %24, ptr @.offload_maptypes.1, ptr null, ptr null)
// CHECK-NEXT:   call void @__tgt_unregister_lib(ptr %EmptyDesc)
// CHECK:        store ptr %x, ptr %0, align 8
// CHECK-NEXT:   call void asm sideeffect "", "r,~{memory}"(ptr nonnull %0)
// CHECK:        ret void
// CHECK-NEXT: }

// CHECK: Function Attrs: nounwind
// CHECK: declare i32 @__tgt_target_kernel(ptr, i64, i32, i32, ptr, ptr)

#[unsafe(no_mangle)]
#[inline(never)]
pub fn kernel_1(x: &mut [f32; 256]) {
    for i in 0..256 {
        x[i] = 21.0;
    }
}
