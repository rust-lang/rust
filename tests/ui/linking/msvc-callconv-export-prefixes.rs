//@ build-pass
//@ add-minicore
//@ revisions: i686 x86_64
//@[i686] compile-flags: --target i686-pc-windows-msvc
//@[x86_64] compile-flags: --target x86_64-pc-windows-msvc
//@ compile-flags: -Clinker=rust-lld -Clink-arg=/NOENTRY -Cpanic=abort
//@ needs-llvm-components: x86
//@ needs-rust-lld
//@ ignore-backends: gcc

#![feature(no_core, abi_vectorcall)]
#![no_core]
#![crate_type = "cdylib"]

extern crate minicore;
use minicore::*;

#[unsafe(export_name = "\u{1}ctl_vectorcall")]
pub extern "vectorcall" fn escaped(_: u32, _: u32) {}

#[unsafe(export_name = "?q_vectorcall")]
pub extern "vectorcall" fn question(_: u32, _: u32) {}

#[unsafe(export_name = "normal_vectorcall")]
pub extern "vectorcall" fn normal(_: u32, _: u32) {}
