// no-system-llvm
// compile-flags: -O
// only-64bit (because the LLVM type of i64 for usize shows up)
// ignore-debug: all the `debug_assert`s get in the way

#![crate_type = "lib"]
#![feature(array_repeat)]
#![feature(array_resize)]

// CHECK-LABEL: @array_repeat_byte
#[no_mangle]
pub fn array_repeat_byte() -> [u8; 1234] {
    // CHECK-NEXT: start:
    // CHECK-NEXT: %1 = getelementptr
    // CHECK-NEXT: tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 1 dereferenceable(1234) %1, i8 42, i64 1234, i1 false)
    // CHECK-NEXT: ret void
    std::array::repeat(42)
}

// CHECK-LABEL: @array_resize_byte_shrink
#[no_mangle]
pub fn array_resize_byte_shrink(a: [u8; 300]) -> [u8; 100] {
    // CHECK-NOT: @llvm.memset
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
    // CHECK-SAME: dereferenceable(100)
    // CHECK-SAME: i64 100
    // CHECK-NOT: @llvm.memset
    a.resize(42)
}

// CHECK-LABEL: @array_resize_byte_grow
#[no_mangle]
pub fn array_resize_byte_grow(a: [u8; 100]) -> [u8; 300] {
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
    // CHECK-SAME: dereferenceable(100)
    // CHECK-SAME: i64 100
    // CHECK: call void @llvm.memset.p0i8.i64
    // CHECK-SAME: dereferenceable(200)
    // CHECK-SAME: i8 42, i64 200
    a.resize(42)
}

// CHECK-LABEL: @array_resize_with_byte_grow
#[no_mangle]
pub fn array_resize_with_byte_grow(a: [u8; 100]) -> [u8; 300] {
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
    // CHECK-SAME: dereferenceable(100)
    // CHECK-SAME: i64 100
    // CHECK: call void @llvm.memset.p0i8.i64
    // CHECK-SAME: dereferenceable(200)
    // CHECK-SAME: i8 42, i64 200
    a.resize_with(|| 42)
}

// CHECK-LABEL: @array_resize_string_moves_value
#[no_mangle]
pub fn array_resize_string_moves_value(a: [String; 1], b: String) -> [String; 2] {
    // CHECK-NOT: __rust_dealloc
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
    // CHECK-SAME: dereferenceable(24)
    // CHECK-SAME: i64 24
    // CHECK-NOT: __rust_dealloc
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
    // CHECK-SAME: dereferenceable(24)
    // CHECK-SAME: i64 24
    // CHECK-NOT: __rust_dealloc
    a.resize(b)
}
