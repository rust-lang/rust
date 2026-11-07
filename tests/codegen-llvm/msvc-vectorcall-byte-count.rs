// Regression test for rust-lang/rust#165299.
// LLVM must not calculate the decoration from an indirect pointer argument.
//
//@ add-minicore
//@ compile-flags: --target x86_64-pc-windows-msvc
//@ needs-llvm-components: x86

#![feature(abi_vectorcall, no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct Big {
    pub data: [u64; 4],
}

// CHECK-DAG: define x86_vectorcallcc void @"\01vc_big@@32"(
#[unsafe(export_name = "vc_big")]
pub extern "vectorcall" fn vc_big(_: Big) {}

// CHECK-DAG: define x86_vectorcallcc void @"\01vc_u64@@8"(
#[unsafe(export_name = "vc_u64")]
pub extern "vectorcall" fn vc_u64(_: u64) {}
