//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// This test is verifying that we generate __tgt_target_data_*_mapper before and after a call to
// __tgt_target_kernel, and initialize all needed variables. It also verifies some related globals.
// Better documentation to what each global or variable means is available in the gpu offload code,
// or the LLVM offload documentation.

#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![no_main]

#[unsafe(no_mangle)]
fn main() {
    let mut x = [3.0; 256];
    kernel_1(&mut x);
    core::hint::black_box(&x);
}

pub fn kernel_1(x: &mut [f32; 256]) {
    core::intrinsics::offload(kernel_1, [256, 1, 1], [32, 1, 1], (x,))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn _kernel_1(x: &mut [f32; 256]) {
    for i in 0..256 {
        x[i] = 21.0;
    }
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }
// CHECK: %struct.__tgt_offload_entry = type { i64, i16, i16, i32, ptr, ptr, i64, i64, ptr }
// CHECK: %struct.__tgt_kernel_arguments = type { i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, [3 x i32], [3 x i32], i32 }

// CHECK: @anon.[[ID:.*]].0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
// CHECK: @anon.{{.*}}.1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @anon.[[ID]].0 }, align 8

// CHECK-DAG: @.omp_offloading.descriptor = internal constant { i32, ptr, ptr, ptr } zeroinitializer
// CHECK-DAG: @llvm.global_ctors = appending constant [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr @.omp_offloading.descriptor_reg, ptr null }]
// CHECK-DAG: @.offload_sizes.[[K:[^ ]*kernel_1]] = private unnamed_addr constant [1 x i64] [i64 1024]
// CHECK-DAG: @.offload_maptypes.[[K]] = private unnamed_addr constant [1 x i64] [i64 35]
// CHECK-DAG: @.[[K]].region_id = internal constant i8 0
// CHECK-DAG: @.offloading.entry_name.[[K]] = internal unnamed_addr constant [{{[0-9]+}} x i8] c"[[K]]{{\\00}}", section ".llvm.rodata.offloading", align 1
// CHECK-DAG: @.offloading.entry.[[K]] = internal constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr @.[[K]].region_id, ptr @.offloading.entry_name.[[K]], i64 0, i64 0, ptr null }, section "llvm_offload_entries", align 8

// CHECK: declare i32 @__tgt_target_kernel(ptr, i64, i32, i32, ptr, ptr)

// CHECK-LABEL: define{{( dso_local)?}} void @main()
// CHECK-NEXT: start:
// CHECK-NEXT:  %0 = alloca [8 x i8], align 8
// CHECK-NEXT:  %x = alloca [1024 x i8], align 16
// CHECK-NEXT:   %.offload_baseptrs = alloca [1 x ptr], align 8
// CHECK-NEXT:   %.offload_ptrs = alloca [1 x ptr], align 8
// CHECK-NEXT:   %.offload_sizes = alloca [1 x i64], align 8
// CHECK-NEXT:   %kernel_args = alloca %struct.__tgt_kernel_arguments, align 8
// CHECK:   %dummy = load volatile ptr, ptr @.offload_sizes.[[K]], align 8
// CHECK-NEXT:   %dummy1 = load volatile ptr, ptr @.offloading.entry.[[K]], align 8
// CHECK-NEXT:   call void @__tgt_init_all_rtls()
// CHECK-NEXT:   store ptr %x, ptr %.offload_baseptrs, align 8
// CHECK-NEXT:   store ptr %x, ptr %.offload_ptrs, align 8
// CHECK-NEXT:   store i64 1024, ptr %.offload_sizes, align 8
// CHECK-NEXT:   call void @__tgt_target_data_begin_mapper(ptr nonnull @anon.{{.*}}.1, i64 -1, i32 1, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr nonnull @.offload_maptypes.[[K]], ptr null, ptr null)
// CHECK-NEXT:   store i32 3, ptr %kernel_args, align 8
// CHECK-NEXT:   [[P4:%[^ ]+]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 4
// CHECK-NEXT:   store i32 1, ptr [[P4]], align 4
// CHECK-NEXT:   [[P8:%[^ ]+]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 8
// CHECK-NEXT:   store ptr %.offload_baseptrs, ptr [[P8]], align 8
// CHECK-NEXT:   [[P16:%[^ ]+]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 16
// CHECK-NEXT:   store ptr %.offload_ptrs, ptr [[P16]], align 8
// CHECK-NEXT:   [[P24:%[^ ]+]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 24
// CHECK-NEXT:   store ptr %.offload_sizes, ptr [[P24]], align 8
// CHECK-NEXT:   [[P32:%[^ ]+]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 32
// CHECK-NEXT:   store ptr @.offload_maptypes.[[K]], ptr [[P32]], align 8
// CHECK-NEXT:   [[P40:%[^ ]+]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 40
// CHECK-NEXT:   [[P72:%[^ ]+]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 72
// CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) [[P40]], i8 0, i64 32, i1 false)
// CHECK-NEXT:   store <4 x i32> <i32 256, i32 1, i32 1, i32 32>, ptr [[P72]], align 8
// CHECK-NEXT:   [[P88:%[^ ]+]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 88
// CHECK-NEXT:   store i32 1, ptr [[P88]], align 8
// CHECK-NEXT:   [[P92:%[^ ]+]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 92
// CHECK-NEXT:   store i32 1, ptr [[P92]], align 4
// CHECK-NEXT:   [[P96:%[^ ]+]] = getelementptr inbounds nuw i8, ptr %kernel_args, i64 96
// CHECK-NEXT:   store i32 0, ptr [[P96]], align 8
// CHECK-NEXT:   {{%[^ ]+}} = call i32 @__tgt_target_kernel(ptr nonnull @anon.{{.*}}.1, i64 -1, i32 256, i32 32, ptr nonnull @.[[K]].region_id, ptr nonnull %kernel_args)
// CHECK-NEXT:   call void @__tgt_target_data_end_mapper(ptr nonnull @anon.{{.*}}.1, i64 -1, i32 1, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr nonnull @.offload_maptypes.[[K]], ptr null, ptr null)
// CHECK:   ret void
// CHECK-NEXT: }

// CHECK: declare void @__tgt_register_lib(ptr) local_unnamed_addr
// CHECK: declare void @__tgt_unregister_lib(ptr) local_unnamed_addr

// CHECK-LABEL: define internal void @.omp_offloading.descriptor_reg() section ".text.startup" {
// CHECK-NEXT: entry:
// CHECK-NEXT:   call void @__tgt_register_lib(ptr nonnull @.omp_offloading.descriptor)
// CHECK-NEXT:   %0 = {{tail }}call i32 @atexit(ptr nonnull @.omp_offloading.descriptor_unreg)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK-LABEL: define internal void @.omp_offloading.descriptor_unreg() section ".text.startup" {
// CHECK-NEXT: entry:
// CHECK-NEXT:   call void @__tgt_unregister_lib(ptr nonnull @.omp_offloading.descriptor)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: !{i32 7, !"openmp", i32 51}
