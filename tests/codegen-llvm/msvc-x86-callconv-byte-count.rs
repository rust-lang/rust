// Regression test for rust-lang/rust#165299.
// Verify decoration using source argument layouts on 32-bit MSVC.
//
//@ add-minicore
//@ compile-flags: --target i686-pc-windows-msvc
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

// CHECK-DAG: @"\01@fast_big@32"
#[unsafe(export_name = "fast_big")]
pub extern "fastcall" fn fast_big(_: Big) {}

// CHECK-DAG: @"\01_std_big@32"
#[unsafe(export_name = "std_big")]
pub extern "stdcall" fn std_big(_: Big) {}

// CHECK-DAG: @"\01vec_big@@32"
#[unsafe(export_name = "vec_big")]
pub extern "vectorcall" fn vec_big(_: Big) {}

// CHECK-DAG: @"\01vec_u64@@8"
#[unsafe(export_name = "vec_u64")]
pub extern "vectorcall" fn vec_u64(_: u64) {}

// Names beginning with '?' are already decorated.
// CHECK-DAG: @"?special"
#[unsafe(export_name = "?special")]
pub extern "stdcall" fn question(_: u32) {}

// The LLVM escape prefix must not acquire another decoration.
// CHECK-DAG: @"\01special"
#[unsafe(export_name = "\u{1}special")]
pub extern "stdcall" fn escaped(_: u32) {}
