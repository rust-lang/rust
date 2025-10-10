//! Test calling variadic functions with various ABIs.
//@ add-core-stubs
//@ compile-flags: -Z merge-functions=disabled
//@ revisions: x86_32 x86_32_win x86_64 aarch64 arm32
//@[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64] needs-llvm-components: x86
//@[x86_32_win] compile-flags: --target i686-pc-windows-msvc
//@[x86_32_win] needs-llvm-components: x86
//@[x86_32] compile-flags: --target i686-unknown-linux-gnu
//@[x86_32] needs-llvm-components: x86
//@[aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64] needs-llvm-components: aarch64
//@[arm32] compile-flags: --target armv7-unknown-linux-gnueabihf
//@[arm32] needs-llvm-components: arm
#![crate_type = "lib"]
#![feature(no_core)]
#![feature(extern_system_varargs)]
#![no_core]

extern crate minicore;

// CHECK-LABEL: @c
#[unsafe(no_mangle)]
fn c(f: extern "C" fn(i32, ...)) {
    // CHECK: call void (i32, ...)
    f(22, 44);
}

// CHECK-LABEL: @system
#[unsafe(no_mangle)]
fn system(f: extern "system" fn(i32, ...)) {
    // Crucially, this is *always* the C calling convention, even on Windows.
    // CHECK: call void (i32, ...)
    f(22, 44);
}

// x86_32-LABEL: @cdecl
#[unsafe(no_mangle)]
#[cfg(target_arch = "x86")]
fn cdecl(f: extern "cdecl" fn(i32, ...)) {
    // x86_32: call void (i32, ...)
    f(22, 44);
}

// x86_64-LABEL: @sysv
#[unsafe(no_mangle)]
#[cfg(target_arch = "x86_64")]
fn sysv(f: extern "sysv64" fn(i32, ...)) {
    // x86_64: call x86_64_sysvcc void (i32, ...)
    f(22, 44);
}

// x86_64-LABEL: @win
#[unsafe(no_mangle)]
#[cfg(target_arch = "x86_64")]
fn win(f: extern "win64" fn(i32, ...)) {
    // x86_64: call win64cc void (i32, ...)
    f(22, 44);
}

// CHECK-LABEL: @efiapi
#[unsafe(no_mangle)]
#[cfg(any(
    target_arch = "arm",
    target_arch = "aarch64",
    target_arch = "riscv32",
    target_arch = "riscv64",
    target_arch = "x86",
    target_arch = "x86_64"
))]
fn efiapi(f: extern "efiapi" fn(i32, ...)) {
    // x86_32: call void (i32, ...)
    // x86_32_win: call void (i32, ...)
    // x86_64: call win64cc void (i32, ...)
    // aarch64: call void (i32, ...)
    // arm32: call arm_aapcscc void (i32, ...)
    f(22, 44);
}

// arm32-LABEL: @aapcs
#[unsafe(no_mangle)]
#[cfg(target_arch = "arm")]
fn aapcs(f: extern "aapcs" fn(i32, ...)) {
    // arm32: call arm_aapcscc void (i32, ...)
    f(22, 44);
}
