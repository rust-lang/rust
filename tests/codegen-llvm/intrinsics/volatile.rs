//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics;

#[repr(align(32))]
pub struct CustomZst;

type UninitFatPointer = std::mem::MaybeUninit<&'static dyn std::fmt::Debug>;

// CHECK-LABEL: @volatile_copy_memory
#[no_mangle]
pub unsafe fn volatile_copy_memory(a: *mut u8, b: *const u8) {
    // CHECK: llvm.memmove.{{\w*(.*true)}}
    intrinsics::volatile_copy_memory(a, b, 1)
}

// CHECK-LABEL: @volatile_copy_nonoverlapping_memory
#[no_mangle]
pub unsafe fn volatile_copy_nonoverlapping_memory(a: *mut u8, b: *const u8) {
    // CHECK: llvm.memcpy.{{\w*(.*true)}}
    intrinsics::volatile_copy_nonoverlapping_memory(a, b, 1)
}

// CHECK-LABEL: @volatile_set_memory
#[no_mangle]
pub unsafe fn volatile_set_memory(a: *mut u8, b: u8) {
    // CHECK: llvm.memset.{{\w*(.*true)}}
    intrinsics::volatile_set_memory(a, b, 1)
}

// CHECK-LABEL: @volatile_load
#[no_mangle]
pub unsafe fn volatile_load(a: *const u16) -> u16 {
    // CHECK: [[TEMP:%.+]] = load volatile i16, ptr %a
    // CHECK-SAME: align 2{{,|$}}
    // CHECK-NEXT: ret i16 [[TEMP]]
    intrinsics::volatile_load(a)
}

// CHECK-LABEL: @volatile_load_bool
#[no_mangle]
pub unsafe fn volatile_load_bool(a: *const bool) -> bool {
    // CHECK: [[TEMP:%.+]] = load volatile i8, ptr %a
    // CHECK-SAME: align 1{{,|$}}
    // CHECK: [[TRUNC:%.+]] = trunc nuw i8 [[TEMP]] to i1
    // CHECK: ret i1 [[TRUNC]]
    intrinsics::volatile_load(a)
}

// CHECK-LABEL: @volatile_load_zst
#[no_mangle]
pub unsafe fn volatile_load_zst(a: *const CustomZst) -> CustomZst {
    // CHECK: start:
    // CHECK-NEXT: ret void
    intrinsics::volatile_load(a)
}

// CHECK-LABEL: @volatile_load_array
// CHECK-SAME: ptr{{.+}}sret([16 x i8]){{.+}}%_0
#[no_mangle]
pub unsafe fn volatile_load_array(a: *const [u16; 8]) -> [u16; 8] {
    // CHECK-NOT: alloca
    // CHECK: [[TEMP:%.+]] = load volatile i128, ptr %a,
    // CHECK-SAME: align 2{{,|$}}
    // CHECK: store i128 [[TEMP]], ptr %_0,
    // CHECK-SAME: align 2{{,|$}}
    // CHECK-NEXT: ret void
    intrinsics::volatile_load(a)
}

// CHECK-LABEL: @volatile_load_fat
#[no_mangle]
pub unsafe fn volatile_load_fat(a: *const UninitFatPointer) -> UninitFatPointer {
    // CHECK: [[ALLOCA:%.+]] = alloca
    // CHECK-SAME: [[SIZE:4|8|16]] x i8
    // CHECK-SAME: align [[ALIGN:2|4|8]]

    // CHECK: [[TEMP:%.+]] = load volatile [[INT:i32|i64|i128]], ptr %a,
    // CHECK-SAME: align [[ALIGN]]{{,|$}}
    // CHECK: store [[INT]] [[TEMP]], ptr [[ALLOCA]],
    // CHECK-SAME: align [[ALIGN]]{{,|$}}

    // CHECK: [[T0:%.+]] = load ptr, ptr [[ALLOCA]], align [[ALIGN]]
    // CHECK: [[T1:%.+]] = getelementptr inbounds i8, ptr [[ALLOCA]]
    // CHECK: [[T2:%.+]] = load ptr, ptr [[T1]], align [[ALIGN]]
    // CHECK: [[P1:%.+]] = insertvalue { ptr, ptr } poison, ptr [[T0]], 0
    // CHECK: [[P2:%.+]] = insertvalue { ptr, ptr } [[P1]], ptr [[T2]], 1
    // CHECK: ret { ptr, ptr } [[P2]]
    intrinsics::volatile_load(a)
}

// CHECK-LABEL: @volatile_store
#[no_mangle]
pub unsafe fn volatile_store(a: *mut u8, b: u8) {
    // CHECK: store volatile
    intrinsics::volatile_store(a, b)
}

// CHECK-LABEL: @unaligned_volatile_load
#[no_mangle]
pub unsafe fn unaligned_volatile_load(a: *const u16) -> u16 {
    // CHECK: [[TEMP:%.+]] = load volatile i16, ptr %a
    // CHECK-SAME: align 1{{,|$}}
    // CHECK-NEXT: ret i16 [[TEMP]]
    intrinsics::unaligned_volatile_load(a)
}

// CHECK-LABEL: @unaligned_volatile_load_bool
#[no_mangle]
pub unsafe fn unaligned_volatile_load_bool(a: *const bool) -> bool {
    // CHECK: [[TEMP:%.+]] = load volatile i8, ptr %a
    // CHECK-SAME: align 1{{,|$}}
    // CHECK: [[TRUNC:%.+]] = trunc nuw i8 [[TEMP]] to i1
    // CHECK: ret i1 [[TRUNC]]
    intrinsics::unaligned_volatile_load(a)
}

// CHECK-LABEL: @unaligned_volatile_load_zst
#[no_mangle]
pub unsafe fn unaligned_volatile_load_zst(a: *const CustomZst) -> CustomZst {
    // CHECK: start:
    // CHECK-NEXT: ret void
    intrinsics::unaligned_volatile_load(a)
}

// CHECK-LABEL: @unaligned_volatile_load_array
// CHECK-SAME: ptr{{.+}}sret([16 x i8]){{.+}}%_0,
#[no_mangle]
pub unsafe fn unaligned_volatile_load_array(a: *const [u16; 8]) -> [u16; 8] {
    // CHECK-NOT: alloca
    // CHECK: [[TEMP:%.+]] = load volatile i128, ptr %a,
    // CHECK-SAME: align 1{{,|$}}
    // CHECK: store i128 [[TEMP]], ptr %_0,
    // CHECK-SAME: align 2{{,|$}}
    // CHECK-NEXT: ret void
    intrinsics::unaligned_volatile_load(a)
}

// CHECK-LABEL: @unaligned_volatile_load_fat
#[no_mangle]
pub unsafe fn unaligned_volatile_load_fat(a: *const UninitFatPointer) -> UninitFatPointer {
    // CHECK: [[ALLOCA:%.+]] = alloca
    // CHECK-SAME: [[SIZE]] x i8
    // CHECK-SAME: align [[ALIGN]]

    // CHECK: [[TEMP:%.+]] = load volatile [[INT]], ptr %a,
    // CHECK-SAME: align 1{{,|$}}
    // CHECK: store [[INT]] [[TEMP]], ptr [[ALLOCA]],
    // CHECK-SAME: align [[ALIGN]]{{,|$}}

    // CHECK: [[T0:%.+]] = load ptr, ptr [[ALLOCA]], align [[ALIGN]]
    // CHECK: [[T1:%.+]] = getelementptr inbounds i8, ptr [[ALLOCA]]
    // CHECK: [[T2:%.+]] = load ptr, ptr [[T1]], align [[ALIGN]]
    // CHECK: [[P1:%.+]] = insertvalue { ptr, ptr } poison, ptr [[T0]], 0
    // CHECK: [[P2:%.+]] = insertvalue { ptr, ptr } [[P1]], ptr [[T2]], 1
    // CHECK: ret { ptr, ptr } [[P2]]
    intrinsics::unaligned_volatile_load(a)
}

// CHECK-LABEL: @unaligned_volatile_store
#[no_mangle]
pub unsafe fn unaligned_volatile_store(a: *mut u8, b: u8) {
    // CHECK: store volatile
    intrinsics::unaligned_volatile_store(a, b)
}
