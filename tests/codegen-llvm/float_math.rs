//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::{
    fadd_algebraic, fadd_fast, fdiv_algebraic, fdiv_fast, fmul_algebraic, fmul_fast,
    frem_algebraic, frem_fast, fsub_algebraic, fsub_fast,
};

// CHECK-LABEL: @add
#[no_mangle]
pub fn add(x: f32, y: f32) -> f32 {
    // CHECK: fadd float
    // CHECK-NOT: fast
    x + y
}

// CHECK-LABEL: @test_fadd_algebraic
#[no_mangle]
pub fn test_fadd_algebraic(x: f32, y: f32) -> f32 {
    // CHECK: fadd reassoc nsz arcp contract float %x, %y
    fadd_algebraic(x, y)
}

// CHECK-LABEL: @test_fsub_algebraic
#[no_mangle]
pub fn test_fsub_algebraic(x: f32, y: f32) -> f32 {
    // CHECK: fsub reassoc nsz arcp contract float %x, %y
    fsub_algebraic(x, y)
}

// CHECK-LABEL: @test_fmul_algebraic
#[no_mangle]
pub fn test_fmul_algebraic(x: f32, y: f32) -> f32 {
    // CHECK: fmul reassoc nsz arcp contract float %x, %y
    fmul_algebraic(x, y)
}

// CHECK-LABEL: @test_fdiv_algebraic
#[no_mangle]
pub fn test_fdiv_algebraic(x: f32, y: f32) -> f32 {
    // CHECK: fdiv reassoc nsz arcp contract float %x, %y
    fdiv_algebraic(x, y)
}

// CHECK-LABEL: @test_frem_algebraic
#[no_mangle]
pub fn test_frem_algebraic(x: f32, y: f32) -> f32 {
    // CHECK: frem reassoc nsz arcp contract float %x, %y
    frem_algebraic(x, y)
}

// CHECK-LABEL: @test_fadd_fast
#[no_mangle]
pub fn test_fadd_fast(x: f32, y: f32) -> f32 {
    // CHECK: fadd fast float %x, %y
    unsafe { fadd_fast(x, y) }
}

// CHECK-LABEL: @test_fsub_fast
#[no_mangle]
pub fn test_fsub_fast(x: f32, y: f32) -> f32 {
    // CHECK: fsub fast float %x, %y
    unsafe { fsub_fast(x, y) }
}

// CHECK-LABEL: @test_fmul_fast
#[no_mangle]
pub fn test_fmul_fast(x: f32, y: f32) -> f32 {
    // CHECK: fmul fast float %x, %y
    unsafe { fmul_fast(x, y) }
}

// CHECK-LABEL: @test_fdiv_fast
#[no_mangle]
pub fn test_fdiv_fast(x: f32, y: f32) -> f32 {
    // CHECK: fdiv fast float %x, %y
    unsafe { fdiv_fast(x, y) }
}

// CHECK-LABEL: @test_frem_fast
#[no_mangle]
pub fn test_frem_fast(x: f32, y: f32) -> f32 {
    // CHECK: frem fast float %x, %y
    unsafe { frem_fast(x, y) }
}
