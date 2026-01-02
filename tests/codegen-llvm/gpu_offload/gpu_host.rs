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

#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![no_main]

#[unsafe(no_mangle)]
fn main() {
    let mut x = [3.0; 256];
    kernel_1(&mut x);
    core::hint::black_box(&x);
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }
// CHECK: %struct.__tgt_offload_entry = type { i64, i16, i16, i32, ptr, ptr, i64, i64, ptr }
// CHECK: %struct.__tgt_bin_desc = type { i32, ptr, ptr, ptr }
// CHECK: %struct.__tgt_kernel_arguments = type { i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, [3 x i32], [3 x i32], i32 }

// CHECK: @anon.{{.*}}.0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
// CHECK: @anon.{{.*}}.1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @anon.{{.*}}.0 }, align 8

// CHECK: @.offload_sizes._kernel_1 = private unnamed_addr constant [1 x i64] [i64 1024]
// CHECK: @.offload_maptypes._kernel_1 = private unnamed_addr constant [1 x i64] [i64 35]
// CHECK: @._kernel_1.region_id = internal constant i8 0
// CHECK: @.offloading.entry_name._kernel_1 = internal unnamed_addr constant [10 x i8] c"_kernel_1\00", section ".llvm.rodata.offloading", align 1
// CHECK: @.offloading.entry._kernel_1 = internal constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr @._kernel_1.region_id, ptr @.offloading.entry_name._kernel_1, i64 0, i64 0, ptr null }, section "llvm_offload_entries", align 8

// CHECK: Function Attrs: nounwind
// CHECK: declare i32 @__tgt_target_kernel(ptr, i64, i32, i32, ptr, ptr)

// CHECK: define{{( dso_local)?}} void @main()
// CHECK-NEXT: start:
// CHECK-NEXT:   %0 = alloca [8 x i8], align 8
// CHECK-NEXT:   %x = alloca [1024 x i8], align 16
// CHECK:        call void @kernel_1(ptr noalias noundef nonnull align 4 dereferenceable(1024) %x)
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %0)
// CHECK-NEXT:   store ptr %x, ptr %0, align 8
// CHECK-NEXT:   call void asm sideeffect "", "r,~{memory}"(ptr nonnull %0)
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %0)
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(i64 1024, ptr nonnull %x)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK:      define{{( dso_local)?}} void @kernel_1(ptr noalias noundef align 4 dereferenceable(1024) %x)
// CHECK-NEXT: start:
// CHECK-NEXT:   %EmptyDesc = alloca %struct.__tgt_bin_desc, align 8
// CHECK-NEXT:   %.offload_baseptrs = alloca [1 x ptr], align 8
// CHECK-NEXT:   %.offload_ptrs = alloca [1 x ptr], align 8
// CHECK-NEXT:   %.offload_sizes = alloca [1 x i64], align 8
// CHECK-NEXT:   %kernel_args = alloca %struct.__tgt_kernel_arguments, align 8
// CHECK-NEXT:   %dummy = load volatile ptr, ptr @.offload_sizes._kernel_1, align 8
// CHECK-NEXT:   %dummy1 = load volatile ptr, ptr @.offloading.entry._kernel_1, align 8
// CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %EmptyDesc, i8 0, i64 32, i1 false)
// CHECK-NEXT:   call void @__tgt_register_lib(ptr nonnull %EmptyDesc)
// CHECK-NEXT:   call void @__tgt_init_all_rtls()
// CHECK-NEXT:   store ptr %x, ptr %.offload_baseptrs, align 8
// CHECK-NEXT:   store ptr %x, ptr %.offload_ptrs, align 8
// CHECK-NEXT:   store i64 1024, ptr %.offload_sizes, align 8
// CHECK-NEXT:   call void @__tgt_target_data_begin_mapper(ptr nonnull @anon.{{.*}}.1, i64 -1, i32 1, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr nonnull @.offload_maptypes._kernel_1, ptr null, ptr null)
// CHECK-NEXT:   store i32 3, ptr %kernel_args, align 8
// CHECK-NEXT:   %0 = getelementptr inbounds nuw i8, ptr %kernel_args, i64 4
// CHECK-NEXT:   store i32 1, ptr %0, align 4
// CHECK-NEXT:   %1 = getelementptr inbounds nuw i8, ptr %kernel_args, i64 8
// CHECK-NEXT:   store ptr %.offload_baseptrs, ptr %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds nuw i8, ptr %kernel_args, i64 16
// CHECK-NEXT:   store ptr %.offload_ptrs, ptr %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds nuw i8, ptr %kernel_args, i64 24
// CHECK-NEXT:   store ptr %.offload_sizes, ptr %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds nuw i8, ptr %kernel_args, i64 32
// CHECK-NEXT:   store ptr @.offload_maptypes._kernel_1, ptr %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds nuw i8, ptr %kernel_args, i64 40
// CHECK-NEXT:   %6 = getelementptr inbounds nuw i8, ptr %kernel_args, i64 72
// CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, i8 0, i64 32, i1 false)
// CHECK-NEXT:   store <4 x i32> <i32 256, i32 1, i32 1, i32 32>, ptr %6, align 8
// CHECK-NEXT:   %.fca.1.gep5 = getelementptr inbounds nuw i8, ptr %kernel_args, i64 88
// CHECK-NEXT:   store i32 1, ptr %.fca.1.gep5, align 8
// CHECK-NEXT:   %.fca.2.gep7 = getelementptr inbounds nuw i8, ptr %kernel_args, i64 92
// CHECK-NEXT:   store i32 1, ptr %.fca.2.gep7, align 4
// CHECK-NEXT:   %7 = getelementptr inbounds nuw i8, ptr %kernel_args, i64 96
// CHECK-NEXT:   store i32 0, ptr %7, align 8
// CHECK-NEXT:   %8 = call i32 @__tgt_target_kernel(ptr nonnull @anon.{{.*}}.1, i64 -1, i32 256, i32 32, ptr nonnull @._kernel_1.region_id, ptr nonnull %kernel_args)
// CHECK-NEXT:   call void @__tgt_target_data_end_mapper(ptr nonnull @anon.{{.*}}.1, i64 -1, i32 1, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr nonnull @.offload_maptypes._kernel_1, ptr null, ptr null)
// CHECK-NEXT:   call void @__tgt_unregister_lib(ptr nonnull %EmptyDesc)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

#[unsafe(no_mangle)]
#[inline(never)]
pub fn kernel_1(x: &mut [f32; 256]) {
    core::intrinsics::offload(_kernel_1, [256, 1, 1], [32, 1, 1], (x,))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn _kernel_1(x: &mut [f32; 256]) {
    for i in 0..256 {
        x[i] = 21.0;
    }
}
