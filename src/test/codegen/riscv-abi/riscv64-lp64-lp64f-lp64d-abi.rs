// compile-flags: --target riscv64gc-unknown-linux-gnu -C no-prepopulate-passes
// needs-llvm-components: riscv

#![crate_type = "lib"]
#![no_core]
#![feature(no_core, lang_items)]
#![allow(improper_ctypes)]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

// CHECK: define void @f_void()
#[no_mangle]
pub extern "C" fn f_void() {}

// CHECK: define noundef zeroext i1 @f_scalar_0(i1 noundef zeroext %a)
#[no_mangle]
pub extern "C" fn f_scalar_0(a: bool) -> bool {
    a
}

// CHECK: define signext i8 @f_scalar_1(i8 signext %x)
#[no_mangle]
pub extern "C" fn f_scalar_1(x: i8) -> i8 {
    x
}

// CHECK: define zeroext i8 @f_scalar_2(i8 zeroext %x)
#[no_mangle]
pub extern "C" fn f_scalar_2(x: u8) -> u8 {
    x
}

// CHECK: define signext i32 @f_scalar_3(i32 signext %x)
#[no_mangle]
pub extern "C" fn f_scalar_3(x: i32) -> u32 {
    x as u32
}

// CHECK: define i64 @f_scalar_4(i64 %x)
#[no_mangle]
pub extern "C" fn f_scalar_4(x: i64) -> i64 {
    x
}

// CHECK: define float @f_fp_scalar_1(float %0)
#[no_mangle]
pub extern "C" fn f_fp_scalar_1(x: f32) -> f32 {
    x
}
// CHECK: define double @f_fp_scalar_2(double %0)
#[no_mangle]
pub extern "C" fn f_fp_scalar_2(x: f64) -> f64 {
    x
}

#[repr(C)]
pub struct Empty {}

// CHECK: define void @f_agg_empty_struct()
#[no_mangle]
pub extern "C" fn f_agg_empty_struct(e: Empty) -> Empty {
    e
}

#[repr(C)]
pub struct Tiny {
    a: u16,
    b: u16,
    c: u16,
    d: u16,
}

// CHECK: define void @f_agg_tiny(i64 %0)
#[no_mangle]
pub extern "C" fn f_agg_tiny(mut e: Tiny) {
}

// CHECK: define i64 @f_agg_tiny_ret()
#[no_mangle]
pub extern "C" fn f_agg_tiny_ret() -> Tiny {
    Tiny { a: 1, b: 2, c: 3, d: 4 }
}

#[repr(C)]
pub struct Small {
    a: i64,
    b: *mut i64,
}

// CHECK: define void @f_agg_small([2 x i64] %0)
#[no_mangle]
pub extern "C" fn f_agg_small(mut x: Small) {
}

// CHECK: define [2 x i64] @f_agg_small_ret()
#[no_mangle]
pub extern "C" fn f_agg_small_ret() -> Small {
    Small { a: 1, b: 0 as *mut _ }
}

#[repr(C)]
pub struct SmallAligned {
    a: i128,
}

// CHECK: define void @f_agg_small_aligned(i128 %0)
#[no_mangle]
pub extern "C" fn f_agg_small_aligned(mut x: SmallAligned) {
}

#[repr(C)]
pub struct Large {
    a: i64,
    b: i64,
    c: i64,
    d: i64,
}

// CHECK: define void @f_agg_large(%Large* {{.*}}%x)
#[no_mangle]
pub extern "C" fn f_agg_large(mut x: Large) {
}

// CHECK: define void @f_agg_large_ret(%Large* {{.*}}sret{{.*}}, i32 signext %i, i8 signext %j)
#[no_mangle]
pub extern "C" fn f_agg_large_ret(i: i32, j: i8) -> Large {
    Large { a: 1, b: 2, c: 3, d: 4 }
}

// CHECK: define void @f_scalar_stack_1(i64 %0, [2 x i64] %1, i128 %2, %Large* {{.*}}%d, i8 zeroext %e, i8 signext %f, i8 %g, i8 %h)
#[no_mangle]
pub extern "C" fn f_scalar_stack_1(
    a: Tiny,
    b: Small,
    c: SmallAligned,
    d: Large,
    e: u8,
    f: i8,
    g: u8,
    h: i8,
) {
}

// CHECK: define void @f_scalar_stack_2(%Large* {{.*}}sret{{.*}} %0, i64 %a, i128 %1, i128 %2, i64 %d, i8 zeroext %e, i8 %f, i8 %g)
#[no_mangle]
pub extern "C" fn f_scalar_stack_2(
    a: u64,
    b: SmallAligned,
    c: SmallAligned,
    d: u64,
    e: u8,
    f: i8,
    g: u8,
) -> Large {
    Large { a: a as i64, b: e as i64, c: f as i64, d: g as i64 }
}

extern "C" {
    fn f_va_callee(_: i32, ...) -> i32;
}

#[no_mangle]
pub unsafe extern "C" fn f_va_caller() {
    // CHECK: call signext i32 (i32, ...) @f_va_callee(i32 signext 1, i32 signext 2, i64 3, double {{.*}}, double {{.*}}, i64 {{.*}}, [2 x i64] {{.*}}, i128 {{.*}}, %Large* {{.*}})
    f_va_callee(
        1,
        2i32,
        3i64,
        4.0f64,
        5.0f64,
        Tiny { a: 1, b: 2, c: 3, d: 4 },
        Small { a: 10, b: 0 as *mut _ },
        SmallAligned { a: 11 },
        Large { a: 12, b: 13, c: 14, d: 15 },
    );
    // CHECK: call signext i32 (i32, ...) @f_va_callee(i32 signext 1, i32 signext 2, i32 signext 3, i32 signext 4, i128 {{.*}}, i32 signext 6, i32 signext 7, i32 8, i32 9)
    f_va_callee(1, 2i32, 3i32, 4i32, SmallAligned { a: 5 }, 6i32, 7i32, 8i32, 9i32);
}
