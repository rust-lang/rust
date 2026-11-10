// Regression test for rust-lang/rust#165299.
//
// Cross-compile a Windows MSVC cdylib and link it to check that the
// linker-visible names agree with the names emitted by LLVM.
//
// `minicore` avoids requiring a Windows target sysroot, while rust-lld
// allows this test to link on non-Windows hosts. `/NOENTRY` avoids needing
// a DLL entry point or Windows startup libraries.
//
// Check normal vectorcall decoration and names explicitly marked as
// already decorated with either `\x01` or `?`.

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

#[unsafe(export_name = "\x01ctl_vectorcall")]
pub extern "vectorcall" fn escaped(_: u32, _: u32) {}

#[unsafe(export_name = "?q_vectorcall")]
pub extern "vectorcall" fn question(_: u32, _: u32) {}

#[unsafe(export_name = "normal_vectorcall")]
pub extern "vectorcall" fn normal(_: u32, _: u32) {}
