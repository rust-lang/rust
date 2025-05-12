//@ add-core-stubs
//@ assembly-output: emit-asm
// ignore-tidy-linelength
//@ revisions: aarch64_apple_darwin
//@ [aarch64_apple_darwin] compile-flags: --target aarch64-apple-darwin
//@ [aarch64_apple_darwin] needs-llvm-components: aarch64
//@ revisions: aarch64_apple_ios
//@ [aarch64_apple_ios] compile-flags: --target aarch64-apple-ios
//@ [aarch64_apple_ios] needs-llvm-components: aarch64
//@ revisions: aarch64_apple_ios_macabi
//@ [aarch64_apple_ios_macabi] compile-flags: --target aarch64-apple-ios-macabi
//@ [aarch64_apple_ios_macabi] needs-llvm-components: aarch64
//@ revisions: aarch64_apple_ios_sim
//@ [aarch64_apple_ios_sim] compile-flags: --target aarch64-apple-ios-sim
//@ [aarch64_apple_ios_sim] needs-llvm-components: aarch64
//@ revisions: aarch64_apple_tvos
//@ [aarch64_apple_tvos] compile-flags: --target aarch64-apple-tvos
//@ [aarch64_apple_tvos] needs-llvm-components: aarch64
//@ revisions: aarch64_apple_tvos_sim
//@ [aarch64_apple_tvos_sim] compile-flags: --target aarch64-apple-tvos-sim
//@ [aarch64_apple_tvos_sim] needs-llvm-components: aarch64
//@ revisions: arm64e_apple_tvos
//@ [arm64e_apple_tvos] compile-flags: --target arm64e-apple-tvos
//@ [arm64e_apple_tvos] needs-llvm-components: aarch64
//@ revisions: aarch64_apple_watchos
//@ [aarch64_apple_watchos] compile-flags: --target aarch64-apple-watchos
//@ [aarch64_apple_watchos] needs-llvm-components: aarch64
//@ revisions: aarch64_apple_watchos_sim
//@ [aarch64_apple_watchos_sim] compile-flags: --target aarch64-apple-watchos-sim
//@ [aarch64_apple_watchos_sim] needs-llvm-components: aarch64
//@ revisions: arm64_32_apple_watchos
//@ [arm64_32_apple_watchos] compile-flags: --target arm64_32-apple-watchos
//@ [arm64_32_apple_watchos] needs-llvm-components: aarch64
//@ revisions: aarch64_apple_visionos
//@ [aarch64_apple_visionos] compile-flags: --target aarch64-apple-visionos
//@ [aarch64_apple_visionos] needs-llvm-components: aarch64
//@ revisions: aarch64_apple_visionos_sim
//@ [aarch64_apple_visionos_sim] compile-flags: --target aarch64-apple-visionos-sim
//@ [aarch64_apple_visionos_sim] needs-llvm-components: aarch64
//@ revisions: arm64e_apple_darwin
//@ [arm64e_apple_darwin] compile-flags: --target arm64e-apple-darwin
//@ [arm64e_apple_darwin] needs-llvm-components: aarch64
//@ revisions: arm64e_apple_ios
//@ [arm64e_apple_ios] compile-flags: --target arm64e-apple-ios
//@ [arm64e_apple_ios] needs-llvm-components: aarch64
//@ revisions: armv7k_apple_watchos
//@ [armv7k_apple_watchos] compile-flags: --target armv7k-apple-watchos
//@ [armv7k_apple_watchos] needs-llvm-components: arm
//@ revisions: armv7s_apple_ios
//@ [armv7s_apple_ios] compile-flags: --target armv7s-apple-ios
//@ [armv7s_apple_ios] needs-llvm-components: arm
//@ revisions: i386_apple_ios
//@ [i386_apple_ios] compile-flags: --target i386-apple-ios
//@ [i386_apple_ios] needs-llvm-components: x86
//@ revisions: i686_apple_darwin
//@ [i686_apple_darwin] compile-flags: --target i686-apple-darwin
//@ [i686_apple_darwin] needs-llvm-components: x86
//@ revisions: x86_64_apple_darwin
//@ [x86_64_apple_darwin] compile-flags: --target x86_64-apple-darwin
//@ [x86_64_apple_darwin] needs-llvm-components: x86
//@ revisions: x86_64_apple_ios
//@ [x86_64_apple_ios] compile-flags: --target x86_64-apple-ios
//@ [x86_64_apple_ios] needs-llvm-components: x86
//@ revisions: x86_64_apple_ios_macabi
//@ [x86_64_apple_ios_macabi] compile-flags: --target x86_64-apple-ios-macabi
//@ [x86_64_apple_ios_macabi] needs-llvm-components: x86
//@ revisions: x86_64_apple_tvos
//@ [x86_64_apple_tvos] compile-flags: --target x86_64-apple-tvos
//@ [x86_64_apple_tvos] needs-llvm-components: x86
//@ revisions: x86_64_apple_watchos_sim
//@ [x86_64_apple_watchos_sim] compile-flags: --target x86_64-apple-watchos-sim
//@ [x86_64_apple_watchos_sim] needs-llvm-components: x86
//@ revisions: x86_64h_apple_darwin
//@ [x86_64h_apple_darwin] compile-flags: --target x86_64h-apple-darwin
//@ [x86_64h_apple_darwin] needs-llvm-components: x86

// Sanity-check that each target can produce assembly code.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// Force linkage to ensure code is actually generated
#[no_mangle]
pub fn test() -> u8 {
    42
}

// CHECK: .section __TEXT,__text
