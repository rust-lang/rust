//@ add-core-stubs
//@ compile-flags: -Z merge-functions=disabled
//@ add-core-stubs

//@ revisions: windows-gnu
//@[windows-gnu] compile-flags: --target x86_64-pc-windows-gnu
//@[windows-gnu] needs-llvm-components: x86

//@ revisions: windows-msvc
//@[windows-msvc] compile-flags: --target x86_64-pc-windows-msvc
//@[windows-msvc] needs-llvm-components: x86

// Also test what happens when using a Windows ABI on Linux.
//@ revisions: linux
//@[linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[linux] needs-llvm-components: x86

#![feature(no_core, rustc_attrs, abi_vectorcall)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// Make sure the argument is always passed when explicitly requesting a Windows ABI.
// Our goal here is to match clang: <https://clang.godbolt.org/z/Wr4jMWq3P>.

// CHECK: define win64cc void @pass_zst_win64(ptr {{[^,]*}})
#[no_mangle]
extern "win64" fn pass_zst_win64(_: ()) {}

// CHECK: define x86_vectorcallcc void @pass_zst_vectorcall(ptr {{[^,]*}})
#[no_mangle]
extern "vectorcall" fn pass_zst_vectorcall(_: ()) {}

// windows-gnu: define void @pass_zst_fastcall(ptr {{[^,]*}})
// windows-msvc: define void @pass_zst_fastcall(ptr {{[^,]*}})
#[no_mangle]
#[cfg(windows)] // "fastcall" is not valid on 64bit Linux
extern "fastcall" fn pass_zst_fastcall(_: ()) {}

// The sysv64 ABI ignores ZST.

// CHECK: define x86_64_sysvcc void @pass_zst_sysv64()
#[no_mangle]
extern "sysv64" fn pass_zst_sysv64(_: ()) {}

// For `extern "C"` functions, ZST are ignored on Linux put passed on Windows.

// linux: define void @pass_zst_c()
// windows-msvc: define void @pass_zst_c(ptr {{[^,]*}})
// windows-gnu: define void @pass_zst_c(ptr {{[^,]*}})
#[no_mangle]
extern "C" fn pass_zst_c(_: ()) {}
