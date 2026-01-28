//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// This test is verifying that we generate __tgt_target_data_*_mapper before and after a call to the
// kernel_1. Better documentation to what each global or variable means is available in the gpu
// offload code, or the LLVM offload documentation.

#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![no_main]

#[unsafe(no_mangle)]
fn main() {
    let mut x = [3.0; 256];
    let y = [1.0; 256];
    kernel_1(&mut x, &y);
    core::hint::black_box(&x);
    core::hint::black_box(&y);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn kernel_1(x: &mut [f32; 256], y: &[f32; 256]) {
    core::intrinsics::offload(_kernel_1, [256, 1, 1], [32, 1, 1], (x, y))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn _kernel_1(x: &mut [f32; 256], y: &[f32; 256]) {
    for i in 0..256 {
        x[i] = 21.0 + y[i];
    }
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }
// CHECK: %struct.__tgt_offload_entry = type { i64, i16, i16, i32, ptr, ptr, i64, i64, ptr }
// CHECK: %struct.__tgt_bin_desc = type { i32, ptr, ptr, ptr }
// CHECK: %struct.__tgt_kernel_arguments = type { i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, [3 x i32], [3 x i32], i32 }

// CHECK: @anon.{{.*}}.0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
// CHECK: @anon.{{.*}}.1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @anon.{{.*}}.0 }, align 8

// CHECK: @.offload_sizes._kernel_1 = private unnamed_addr constant [2 x i64] [i64 1024, i64 1024]
// CHECK: @.offload_maptypes._kernel_1.begin = private unnamed_addr constant [2 x i64] [i64 1, i64 1]
// CHECK: @.offload_maptypes._kernel_1.kernel = private unnamed_addr constant [2 x i64] [i64 32, i64 32]
// CHECK: @.offload_maptypes._kernel_1.end = private unnamed_addr constant [2 x i64] [i64 2, i64 0]
// CHECK: @._kernel_1.region_id = internal constant i8 0
// CHECK: @.offloading.entry_name._kernel_1 = internal unnamed_addr constant [10 x i8] c"_kernel_1\00", section ".llvm.rodata.offloading", align 1
// CHECK: @.offloading.entry._kernel_1 = internal constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr @._kernel_1.region_id, ptr @.offloading.entry_name._kernel_1, i64 0, i64 0, ptr null }, section "llvm_offload_entries", align 8

// CHECK: declare i32 @__tgt_target_kernel(ptr, i64, i32, i32, ptr, ptr)
// CHECK: declare void @__tgt_register_lib(ptr) local_unnamed_addr
// CHECK: declare void @__tgt_unregister_lib(ptr) local_unnamed_addr

// CHECK: define{{( dso_local)?}} void @main()
// CHECK-NEXT: start:
// CHECK-NEXT:   %0 = alloca [8 x i8], align 8
// CHECK-NEXT:   %1 = alloca [8 x i8], align 8
// CHECK-NEXT:   %y = alloca [1024 x i8], align 16
// CHECK-NEXT:   %x = alloca [1024 x i8], align 16
// CHECK:        call void @kernel_1(ptr {{.*}} %x, ptr {{.*}} %y)
// CHECK:   store ptr %x, ptr %1, align 8
// CHECK:   call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1)
// CHECK:   store ptr %y, ptr %0, align 8
// CHECK:   call void asm sideeffect "", "r,~{memory}"(ptr nonnull %0)
// CHECK:   ret void
// CHECK-NEXT: }

// CHECK:      define{{( dso_local)?}} void @kernel_1(ptr noalias noundef align 4 dereferenceable(1024) %x, ptr noalias noundef readonly align 4 captures(address, read_provenance) dereferenceable(1024) %y)
// CHECK-NEXT: start:
// CHECK-NEXT:   %EmptyDesc = alloca %struct.__tgt_bin_desc, align 8
// CHECK-NEXT:   %.offload_baseptrs = alloca [2 x ptr], align 8
// CHECK-NEXT:   %.offload_ptrs = alloca [2 x ptr], align 8
// CHECK-NEXT:   %.offload_sizes = alloca [2 x i64], align 8
// CHECK-NEXT:   %kernel_args = alloca %struct.__tgt_kernel_arguments, align 8
// CHECK-NEXT:   %dummy = load volatile ptr, ptr @.offload_sizes._kernel_1, align 8
// CHECK-NEXT:   %dummy1 = load volatile ptr, ptr @.offloading.entry._kernel_1, align 8
// CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %EmptyDesc, i8 0, i64 32, i1 false)
// CHECK-NEXT:   call void @__tgt_register_lib(ptr nonnull %EmptyDesc)
// CHECK-NEXT:   call void @__tgt_init_all_rtls()
// CHECK-NEXT:   store ptr %x, ptr %.offload_baseptrs, align 8
// CHECK-NEXT:   store ptr %x, ptr %.offload_ptrs, align 8
// CHECK-NEXT:   store i64 1024, ptr %.offload_sizes, align 8
// CHECK-NEXT:   [[GEP_BPTRS_0:%.*]] = getelementptr inbounds nuw i8, ptr %.offload_baseptrs, i64 8
// CHECK-NEXT:   store ptr %y, ptr [[GEP_BPTRS_0]], align 8
// CHECK-NEXT:   [[GEP_PTRS_1:%.*]] = getelementptr inbounds nuw i8, ptr %.offload_ptrs, i64 8
// CHECK-NEXT:   store ptr %y, ptr [[GEP_PTRS_1]], align 8
// CHECK-NEXT:   [[GEP_SIZES_1:%.*]] = getelementptr inbounds nuw i8, ptr %.offload_sizes, i64 8
// CHECK-NEXT:   store i64 1024, ptr [[GEP_SIZES_1]], align 8
// CHECK-NEXT:   call void @__tgt_target_data_begin_mapper(ptr nonnull @anon.{{.*}}.1, i64 -1, i32 2, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr nonnull @.offload_maptypes._kernel_1.begin, ptr null, ptr null)
// CHECK-NEXT:   store i32 3, ptr %kernel_args, align 8
// CHECK-NEXT:   [[KARGS_OFF4:%.*]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 4
// CHECK-NEXT:   store i32 2, ptr [[KARGS_OFF4]], align 4
// CHECK-NEXT:   [[KARGS_OFF8:%.*]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 8
// CHECK-NEXT:   store ptr %.offload_baseptrs, ptr [[KARGS_OFF8]], align 8
// CHECK-NEXT:   [[KARGS_OFF16:%.*]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 16
// CHECK-NEXT:   store ptr %.offload_ptrs, ptr [[KARGS_OFF16]], align 8
// CHECK-NEXT:   [[KARGS_OFF24:%.*]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 24
// CHECK-NEXT:   store ptr %.offload_sizes, ptr [[KARGS_OFF24]], align 8
// CHECK-NEXT:   [[KARGS_OFF32:%.*]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 32
// CHECK-NEXT:   store ptr @.offload_maptypes._kernel_1.kernel, ptr [[KARGS_OFF32]], align 8
// CHECK-NEXT:   [[KARGS_OFF40:%.*]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 40
// CHECK-NEXT:   [[KARGS_OFF72:%.*]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 72
// CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) [[KARGS_OFF40]], i8 0, i64 32, i1 false)
// CHECK-NEXT:   store <4 x i32> <i32 256, i32 1, i32 1, i32 32>, ptr [[KARGS_OFF72]], align 8
// CHECK-NEXT:   %.fca.1.gep5 = getelementptr inbounds nuw i8, ptr %kernel_args, i64 88
// CHECK-NEXT:   store i32 1, ptr %.fca.1.gep5, align 8
// CHECK-NEXT:   %.fca.2.gep7 = getelementptr inbounds nuw i8, ptr %kernel_args, i64 92
// CHECK-NEXT:   store i32 1, ptr %.fca.2.gep7, align 4
// CHECK-NEXT:   [[KARGS_OFF96:%.*]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 96
// CHECK-NEXT:   store i32 0, ptr [[KARGS_OFF96]], align 8
// CHECK-NEXT:   [[TGT_RET:%.*]] = call i32 @__tgt_target_kernel(ptr nonnull @anon.{{.*}}.1, i64 -1, i32 256, i32 32, ptr nonnull @._kernel_1.region_id, ptr nonnull %kernel_args)
// CHECK-NEXT:   call void @__tgt_target_data_end_mapper(ptr nonnull @anon.{{.*}}.1, i64 -1, i32 2, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr nonnull @.offload_maptypes._kernel_1.end, ptr null, ptr null)
// CHECK-NEXT:  call void @__tgt_unregister_lib(ptr nonnull %EmptyDesc)
// CHECK-NEXT:  ret void
// CHECK-NEXT: }

// CHECK: !{i32 7, !"openmp", i32 51}
