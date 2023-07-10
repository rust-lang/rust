// compile-flags: -C opt-level=0 -C no-prepopulate-passes
// min-llvm-version: 15.0 # this test assumes `ptr`s and thus no `pointercast`s

#![crate_type = "lib"]

// With opaque ptrs in LLVM, `transmute` can load/store any `alloca` as any type,
// without needing to pointercast, and SRoA will turn that into a `bitcast`.
// Thus memory-to-memory transmutes don't need to generate them ourselves.

// However, `bitcast`s and `ptrtoint`s and `inttoptr`s are still worth doing when
// that allows us to avoid the `alloca`s entirely; see `rvalue_creates_operand`.

// CHECK-LABEL: define{{.*}}i32 @f32_to_bits(float %x)
// CHECK: %_0 = bitcast float %x to i32
// CHECK-NEXT: ret i32 %_0
#[no_mangle]
pub fn f32_to_bits(x: f32) -> u32 {
    unsafe { std::mem::transmute(x) }
}

// CHECK-LABEL: define{{.*}}i8 @bool_to_byte(i1 zeroext %b)
// CHECK: %_0 = zext i1 %b to i8
// CHECK-NEXT: ret i8 %_0
#[no_mangle]
pub fn bool_to_byte(b: bool) -> u8 {
    unsafe { std::mem::transmute(b) }
}

// CHECK-LABEL: define{{.*}}zeroext i1 @byte_to_bool(i8 %byte)
// CHECK: %_0 = trunc i8 %byte to i1
// CHECK-NEXT: ret i1 %_0
#[no_mangle]
pub unsafe fn byte_to_bool(byte: u8) -> bool {
    std::mem::transmute(byte)
}

// CHECK-LABEL: define{{.*}}ptr @ptr_to_ptr(ptr %p)
// CHECK: ret ptr %p
#[no_mangle]
pub fn ptr_to_ptr(p: *mut u16) -> *mut u8 {
    unsafe { std::mem::transmute(p) }
}

// CHECK: define{{.*}}[[USIZE:i[0-9]+]] @ptr_to_int(ptr %p)
// CHECK: %_0 = ptrtoint ptr %p to [[USIZE]]
// CHECK-NEXT: ret [[USIZE]] %_0
#[no_mangle]
pub fn ptr_to_int(p: *mut u16) -> usize {
    unsafe { std::mem::transmute(p) }
}

// CHECK: define{{.*}}ptr @int_to_ptr([[USIZE]] %i)
// CHECK: %_0 = inttoptr [[USIZE]] %i to ptr
// CHECK-NEXT: ret ptr %_0
#[no_mangle]
pub fn int_to_ptr(i: usize) -> *mut u16 {
    unsafe { std::mem::transmute(i) }
}
