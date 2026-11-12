//@ add-minicore
//@ compile-flags: -Z merge-functions=disabled

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

#[repr(C)]
struct CMaybeZst;

#[repr(transparent)]
struct CMaybeZst2((), CMaybeZst, ());

// Make sure the argument is always passed when explicitly requesting a Windows ABI,
// and it is `repr(C)` - but not if it is `repr(Rust)`.
// Our goal here is to match clang: <https://clang.godbolt.org/z/Wr4jMWq3P>.

// CHECK: define win64cc void @pass_rust_zst_win64()
#[no_mangle]
extern "win64" fn pass_rust_zst_win64(_: ()) {}

// CHECK: define win64cc void @pass_c_maybezst_win64(ptr {{[^,]*}})
#[no_mangle]
extern "win64" fn pass_c_maybezst_win64(_: CMaybeZst) {}

// CHECK: define win64cc void @pass_c_maybezst_2_win64(ptr {{[^,]*}})
#[no_mangle]
extern "win64" fn pass_c_maybezst_2_win64(_: CMaybeZst2) {}

// CHECK: define x86_vectorcallcc void @pass_rust_zst_vectorcall()
#[no_mangle]
extern "vectorcall" fn pass_rust_zst_vectorcall(_: ()) {}

// CHECK: define x86_vectorcallcc void @pass_c_maybezst_vectorcall(ptr {{[^,]*}})
#[no_mangle]
extern "vectorcall" fn pass_c_maybezst_vectorcall(_: CMaybeZst) {}

// CHECK: define x86_vectorcallcc void @pass_c_maybezst_2_vectorcall(ptr {{[^,]*}})
#[no_mangle]
extern "vectorcall" fn pass_c_maybezst_2_vectorcall(_: CMaybeZst2) {}

// windows-gnu: define void @pass_rust_zst_fastcall()
// windows-msvc: define void @pass_rust_zst_fastcall()
#[no_mangle]
#[cfg(windows)] // "fastcall" is not valid on 64bit Linux
extern "fastcall" fn pass_rust_zst_fastcall(_: ()) {}

// windows-gnu: define void @pass_c_maybezst_fastcall(ptr {{[^,]*}})
// windows-msvc: define void @pass_c_maybezst_fastcall(ptr {{[^,]*}})
#[no_mangle]
#[cfg(windows)] // "fastcall" is not valid on 64bit Linux
extern "fastcall" fn pass_c_maybezst_fastcall(_: CMaybeZst) {}

// windows-gnu: define void @pass_c_maybezst_2_fastcall(ptr {{[^,]*}})
// windows-msvc: define void @pass_c_maybezst_2_fastcall(ptr {{[^,]*}})
#[no_mangle]
#[cfg(windows)] // "fastcall" is not valid on 64bit Linux
extern "fastcall" fn pass_c_maybezst_2_fastcall(_: CMaybeZst2) {}

// The sysv64 ABI ignores ZST.

// CHECK: define x86_64_sysvcc void @pass_rust_zst_sysv64()
#[no_mangle]
extern "sysv64" fn pass_rust_zst_sysv64(_: ()) {}

// CHECK: define x86_64_sysvcc void @pass_c_maybezst_sysv64()
#[no_mangle]
extern "sysv64" fn pass_c_maybezst_sysv64(_: CMaybeZst) {}

// CHECK: define x86_64_sysvcc void @pass_c_maybezst_2_sysv64()
#[no_mangle]
extern "sysv64" fn pass_c_maybezst_2_sysv64(_: CMaybeZst2) {}

// For `extern "C"` functions, ZST are ignored on Linux put passed on Windows.

// linux: define void @pass_rust_zst_c()
// windows-msvc: define void @pass_rust_zst_c()
// windows-gnu: define void @pass_rust_zst_c()
#[no_mangle]
extern "C" fn pass_rust_zst_c(_: ()) {}

// linux: define void @pass_c_maybezst_c()
// windows-msvc: define void @pass_c_maybezst_c(ptr {{[^,]*}})
// windows-gnu: define void @pass_c_maybezst_c(ptr {{[^,]*}})
#[no_mangle]
extern "C" fn pass_c_maybezst_c(_: CMaybeZst) {}

// linux: define void @pass_c_maybezst_2_c()
// windows-msvc: define void @pass_c_maybezst_2_c(ptr {{[^,]*}})
// windows-gnu: define void @pass_c_maybezst_2_c(ptr {{[^,]*}})
#[no_mangle]
extern "C" fn pass_c_maybezst_2_c(_: CMaybeZst2) {}

// Now check `repr(C)` return types.
// Again, we seek to match clang: <https://clang.godbolt.org/z/hKv74ThnE>

// CHECK: define win64cc void @ret_c_maybezst_win64(ptr {{[^,]*}})
#[no_mangle]
extern "win64" fn ret_c_maybezst_win64() -> CMaybeZst {
    CMaybeZst
}

// CHECK: define x86_vectorcallcc void @ret_c_maybezst_vectorcall(ptr {{[^,]*}})
#[no_mangle]
extern "vectorcall" fn ret_c_maybezst_vectorcall() -> CMaybeZst {
    CMaybeZst
}

// windows-gnu: define void @ret_c_maybezst_fastcall(ptr {{[^,]*}})
// windows-msvc: define void @ret_c_maybezst_fastcall(ptr {{[^,]*}})
#[no_mangle]
#[cfg(windows)] // "fastcall" is not valid on 64bit Linux
extern "fastcall" fn ret_c_maybezst_fastcall() -> CMaybeZst {
    CMaybeZst
}

// The sysv64 ABI ignores ZST.

// CHECK: define x86_64_sysvcc void @ret_c_maybezst_sysv64()
#[no_mangle]
extern "sysv64" fn ret_c_maybezst_sysv64() -> CMaybeZst {
    CMaybeZst
}

// For `extern "C"` functions, ZST are ignored on Linux but returned via pointer on Windows.

// linux: define void @ret_c_maybezst_c()
// windows-msvc: define void @ret_c_maybezst_c(ptr {{[^,]*}})
// windows-gnu: define void @ret_c_maybezst_c(ptr {{[^,]*}})
#[no_mangle]
extern "C" fn ret_c_maybezst_c() -> CMaybeZst {
    CMaybeZst
}
