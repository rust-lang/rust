//@ add-minicore
//@ compile-flags: -Copt-level=3 --target hexagon-unknown-linux-musl
//@ needs-llvm-components: hexagon
#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct Pair {
    x: i32,
    y: i32,
}

#[repr(C)]
pub struct Bytes {
    a: i8,
    b: i8,
}

// CHECK-LABEL: @pair_in_regs(
// CHECK-SAME: i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i64 %0)
#[no_mangle]
pub extern "C" fn pair_in_regs(a: i32, b: i32, c: i32, d: i32, p: Pair) {}

// CHECK-LABEL: @pair_no_pair_reg(
// CHECK-SAME: ptr {{.*}}byval([8 x i8]) align 4 {{.*}}%p)
#[no_mangle]
pub extern "C" fn pair_no_pair_reg(a: i32, b: i32, c: i32, d: i32, e: i32, p: Pair) {}

// CHECK-LABEL: @pair_on_stack(
// CHECK-SAME: ptr {{.*}}byval([8 x i8]) align 4 {{.*}}%p)
#[no_mangle]
pub extern "C" fn pair_on_stack(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, p: Pair) {}

// CHECK-LABEL: @bytes_in_reg(
// CHECK-SAME: i32 noundef %e, i16 %0, i32 noundef %x)
#[no_mangle]
pub extern "C" fn bytes_in_reg(a: i32, b: i32, c: i32, d: i32, e: i32, s: Bytes, x: i32) {}

// CHECK-LABEL: @bytes_on_stack(
// CHECK-SAME: ptr {{.*}}byval([2 x i8]) align 1 {{.*}}%s, i32 noundef %x)
#[no_mangle]
pub extern "C" fn bytes_on_stack(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, s: Bytes, x: i32) {
}

// CHECK-LABEL: define{{.*}} i16 @ret_bytes(
#[no_mangle]
pub extern "C" fn ret_bytes() -> Bytes {
    Bytes { a: 1, b: 2 }
}

// CHECK-LABEL: @va_list_then_bytes(
// CHECK-SAME: i32 noundef %e, ptr {{.*}}%va, ptr {{.*}}byval([2 x i8]) align 1 {{.*}}%s, ptr {{.*}}byval([2 x i8]) align 1 {{.*}}%t)
#[no_mangle]
pub extern "C" fn va_list_then_bytes(
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    e: i32,
    va: ffi::VaList<'_>,
    s: Bytes,
    t: Bytes,
) {
}
