//@ add-minicore
//@ build-fail
//@ compile-flags: --target aarch64-unknown-linux-gnu --crate-type lib
//@ needs-llvm-components: aarch64
//@ ignore-backends: gcc
//@ revisions: num_params_too_few num_params_zero num_params_too_many
//@ revisions: vec_element one_nth_not_vec one_nth_not_div
//@ revisions: extend_not_int extend_too_wide
//@ revisions: trunc_not_int trunc_i1 trunc_odd
//@ revisions: sub2_not_vec sub2_ptr sub2_odd sub4_not_div
//@ revisions: bitcast_not_vec bitcast_ptr
//@ revisions: sig_valid_fail
//@[sig_valid_fail] normalize-stderr: "(invalid type parameters for intrinsic `[^`]+`: ).*" -> "${1}$$LLVM_ERROR"

#![feature(no_core, link_llvm_intrinsics)]
#![no_core]

extern crate minicore;
use minicore::*;

// --- NumParamMismatch ---

#[cfg(num_params_too_few)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ucmp.i32"]
    fn foo();
    //[num_params_too_few]~^ ERROR: incorrect number of type parameters in intrinsic `llvm.ucmp.i32`: expected 2, found 1
}

#[cfg(num_params_zero)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop"]
    fn foo();
    //[num_params_zero]~^ ERROR: incorrect number of type parameters in intrinsic `llvm.ctpop`: expected 1, found 0
}

#[cfg(num_params_too_many)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.i32.i32"]
    fn foo();
    //[num_params_too_many]~^ ERROR: incorrect number of type parameters in intrinsic `llvm.ctpop.i32.i32`: expected 1, found 2
}

// --- VecElement ---

#[cfg(vec_element)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.vector.reduce.add.i32"]
    fn foo();
    //[vec_element]~^ ERROR: invalid type parameter `i32` at position 0 in intrinsic `llvm.vector.reduce.add.i32`
}

// --- OneNthEltsVec ---

#[cfg(one_nth_not_vec)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.vector.deinterleave3.i32"]
    fn foo();
    //[one_nth_not_vec]~^ ERROR: invalid type parameter `i32` at position 0 in intrinsic `llvm.vector.deinterleave3.i32`
}

#[cfg(one_nth_not_div)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.vector.deinterleave3.v4i32"]
    fn foo();
    //[one_nth_not_div]~^ ERROR: invalid type parameter `<4 x i32>` at position 0 in intrinsic `llvm.vector.deinterleave3.v4i32`
}

// --- Extend ---

#[cfg(extend_not_int)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.aarch64.neon.sqxtn.f32"]
    fn foo();
    //[extend_not_int]~^ ERROR: invalid type parameter `float` at position 0 in intrinsic `llvm.aarch64.neon.sqxtn.f32`
}

#[cfg(extend_too_wide)]
extern "llvm-intrinsic" {
    // IntegerType::MAX_INT_BITS is 8388607, so 4194304 * 2 overflows it
    #[link_name = "llvm.aarch64.neon.sqxtn.i4194305"]
    fn foo();
    //[extend_too_wide]~^ ERROR: invalid type parameter `i4194305` at position 0 in intrinsic `llvm.aarch64.neon.sqxtn.i4194305`
}

// --- Trunc ---

#[cfg(trunc_not_int)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.aarch64.neon.smull.v4f32"]
    fn foo();
    //[trunc_not_int]~^ ERROR: invalid type parameter `<4 x float>` at position 0 in intrinsic `llvm.aarch64.neon.smull.v4f32`
}

#[cfg(trunc_i1)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.aarch64.neon.smull.v4i1"]
    fn foo();
    //[trunc_i1]~^ ERROR: invalid type parameter `<4 x i1>` at position 0 in intrinsic `llvm.aarch64.neon.smull.v4i1`
}

#[cfg(trunc_odd)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.aarch64.neon.smull.v4i3"]
    fn foo();
    //[trunc_odd]~^ ERROR: invalid type parameter `<4 x i3>` at position 0 in intrinsic `llvm.aarch64.neon.smull.v4i3`
}

// --- Subdivide2 / Subdivide4 ---

#[cfg(sub2_not_vec)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.aarch64.sve.sunpklo.i32"]
    fn foo();
    //[sub2_not_vec]~^ ERROR: invalid type parameter `i32` at position 0 in intrinsic `llvm.aarch64.sve.sunpklo.i32`
}

#[cfg(sub2_ptr)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.aarch64.sve.sunpklo.nxv4p0"]
    fn foo();
    //[sub2_ptr]~^ ERROR: invalid type parameter `<vscale x 4 x ptr>` at position 0 in intrinsic `llvm.aarch64.sve.sunpklo.nxv4p0`
}

#[cfg(sub2_odd)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.aarch64.sve.sunpklo.nxv4i1"]
    fn foo();
    //[sub2_odd]~^ ERROR: invalid type parameter `<vscale x 4 x i1>` at position 0 in intrinsic `llvm.aarch64.sve.sunpklo.nxv4i1`
}

#[cfg(sub4_not_div)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.aarch64.sve.sdot.nxv4i2"]
    fn foo();
    //[sub4_not_div]~^ ERROR: invalid type parameter `<vscale x 4 x i2>` at position 0 in intrinsic `llvm.aarch64.sve.sdot.nxv4i2`
}

// --- VecOfBitcastsToInt ---

#[cfg(bitcast_not_vec)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.aarch64.sve.fexpa.x.i32"]
    fn foo();
    //[bitcast_not_vec]~^ ERROR: invalid type parameter `i32` at position 0 in intrinsic `llvm.aarch64.sve.fexpa.x.i32`
}

#[cfg(bitcast_ptr)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.aarch64.sve.fexpa.x.v4p0"]
    fn foo();
    //[bitcast_ptr]~^ ERROR: invalid type parameter `<4 x ptr>` at position 0 in intrinsic `llvm.aarch64.sve.fexpa.x.v4p0`
}

// --- isSignatureValid (passes early check, fails LLVM constraint check) ---

#[cfg(sig_valid_fail)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.sqrt.i32"]
    fn foo();
    //[sig_valid_fail]~^ ERROR: invalid type parameters for intrinsic `llvm.sqrt.i32`:
}

pub fn test() {
    unsafe { foo() }
}
