// compile-flags: -C no-prepopulate-passes
// min-llvm-version: 15.0 (for opaque pointers)
#![crate_type = "lib"]
#![feature(explicit_tail_calls)]

/// Something that is likely to be passed indirectly
#[repr(C)]
pub struct IndirectProbably {
    _0: u8,
    _1: u32,
    _2: u64,
    _3: u8,
    _4: [u32; 8],
    _5: u8,
    _6: u128,
}


#[no_mangle]
// CHECK-LABEL: @simple_f(
pub fn simple_f() -> u32 {
    // CHECK: %0 = musttail call noundef i32 @simple_g()
    // CHECK: ret i32 %0
    become simple_g();
}

#[no_mangle]
fn simple_g() -> u32 {
    0
}


#[no_mangle]
// CHECK-LABEL: @unit_f(
fn unit_f() {
    // CHECK: musttail call void @unit_g()
    // CHECK: ret void
    become unit_g();
}

#[no_mangle]
fn unit_g() {}


#[no_mangle]
// CHECK-LABEL: @indirect_f(
fn indirect_f() -> IndirectProbably {
    // CHECK: musttail call void @indirect_g(ptr noalias nocapture noundef sret(%IndirectProbably) dereferenceable(80) %0)
    // CHECK: ret void
    become indirect_g();
}

#[no_mangle]
fn indirect_g() -> IndirectProbably {
    todo!()
}


#[no_mangle]
// CHECK-LABEL: @pair_f(
pub fn pair_f() -> (u32, u8) {
    // CHECK: %0 = musttail call { i32, i8 } @pair_g()
    // CHECK: ret { i32, i8 } %0
    become pair_g()
}

#[no_mangle]
fn pair_g() -> (u32, u8) {
    (1, 2)
}


#[no_mangle]
/// Does `src + dst` in a recursive way
// CHECK-LABEL: @flow(
fn flow(src: u64, dst: u64) -> u64 {
    match src {
        0 => dst,
        // CHECK: %1 = musttail call noundef i64 @flow(
        // CHECK: ret i64 %1
        _ => become flow(src - 1, dst + 1),
    }
}

#[no_mangle]
// CHECK-LABEL: @halt(
pub fn halt() -> ! {
    // CHECK: musttail call void @halt()
    // CHECK: ret void
    become halt();
}
