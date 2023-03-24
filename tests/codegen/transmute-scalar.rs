// compile-flags: -O -C no-prepopulate-passes
// min-llvm-version: 15.0 # this test assumes `ptr`s and thus no `pointercast`s

#![crate_type = "lib"]

// With opaque ptrs in LLVM, `transmute` can load/store any `alloca` as any type,
// without needing to pointercast, and SRoA will turn that into a `bitcast`.
// Thus for place-to-place transmutes, there's no point in bothering to attempt
// to `bitcast` the values, and we can just write them directly.

// FIXME: That said, `bitcast`s could still be a valuable addition if they could
// be done in `rvalue_creates_operand`, and thereby avoid the `alloca`s entirely.

// CHECK-LABEL: define{{.*}}i32 @f32_to_bits(float noundef %x)
// CHECK: store float %{{.*}}, ptr %0
// CHECK-NEXT: %[[RES:.*]] = load i32, ptr %0
// CHECK: ret i32 %[[RES]]
#[no_mangle]
pub fn f32_to_bits(x: f32) -> u32 {
    unsafe { std::mem::transmute(x) }
}

// CHECK-LABEL: define{{.*}}i8 @bool_to_byte(i1 noundef zeroext %b)
// CHECK: %1 = zext i1 %b to i8
// CHECK-NEXT: store i8 %1, ptr %0
// CHECK-NEXT: %2 = load i8, ptr %0
// CHECK: ret i8 %2
#[no_mangle]
pub fn bool_to_byte(b: bool) -> u8 {
    unsafe { std::mem::transmute(b) }
}

// CHECK-LABEL: define{{.*}}noundef zeroext i1 @byte_to_bool(i8 noundef %byte)
// CHECK: store i8 %byte, ptr %0
// CHECK-NEXT: %1 = load i8, ptr %0
// CHECK-NEXT: %2 = trunc i8 %1 to i1
// CHECK: ret i1 %2
#[no_mangle]
pub unsafe fn byte_to_bool(byte: u8) -> bool {
    std::mem::transmute(byte)
}

// CHECK-LABEL: define{{.*}}ptr @ptr_to_ptr(ptr noundef %p)
// CHECK: start
// CHECK-NEXT: ret ptr %p
#[no_mangle]
pub fn ptr_to_ptr(p: *mut u16) -> *mut u8 {
    unsafe { std::mem::transmute(p) }
}

// CHECK: define{{.*}}[[USIZE:i[0-9]+]] @ptr_to_int(ptr noundef %p)
// CHECK: store ptr %p, {{.*}}
// CHECK-NEXT: %[[RES:.*]] = load [[USIZE]], ptr %0
// CHECK: ret [[USIZE]] %[[RES]]
#[no_mangle]
pub fn ptr_to_int(p: *mut u16) -> usize {
    unsafe { std::mem::transmute(p) }
}

// CHECK: define{{.*}}ptr @int_to_ptr([[USIZE]] noundef %i)
// CHECK: store [[USIZE]] %i, {{.*}}
// CHECK-NEXT: %[[RES:.*]] = load ptr, {{.*}} %0
// CHECK: ret ptr %[[RES]]
#[no_mangle]
pub fn int_to_ptr(i: usize) -> *mut u16 {
    unsafe { std::mem::transmute(i) }
}
