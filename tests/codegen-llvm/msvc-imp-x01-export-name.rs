//! Verify that `__imp_<symbol>` aliases are built from the undecorated name: no copied `\x01` and
//! no extra underscore on x86.

//@ add-minicore
//@ revisions: x86_64 x86
//@[x86_64] compile-flags: --target x86_64-pc-windows-msvc
//@[x86_64] needs-llvm-components: x86
//@[x86] compile-flags: --target i686-pc-windows-msvc
//@[x86] needs-llvm-components: x86

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// CHECK: @"\01marked_static" ={{.*}}constant [1 x i8] zeroinitializer
#[unsafe(export_name = "\x01marked_static")]
pub static MARKED_STATIC: u8 = 0;

// CHECK: @"\01__imp_marked_static" = global ptr @"\01marked_static"
