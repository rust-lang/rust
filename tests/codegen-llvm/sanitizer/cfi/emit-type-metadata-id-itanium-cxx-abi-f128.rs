// Verifies `f128` maps to `long double` (code `e`) where that's 128-bit
// (aarch64 Linux) and to `_Float128` (code `g`) elsewhere (x86_64, and aarch64
// Apple where `long double` is 64-bit).
//
//@ add-minicore
//@ needs-sanitizer-cfi
//@ revisions: aarch64 x64 darwin
//@ [aarch64] needs-llvm-components: aarch64
//@ [aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@ [x64] needs-llvm-components: x86
//@ [x64] compile-flags: --target x86_64-unknown-linux-gnu
//@ [darwin] needs-llvm-components: aarch64
//@ [darwin] compile-flags: --target aarch64-apple-darwin
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0 -C link-dead-code
//@ compile-flags: -Clto -Zsanitizer=cfi -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer
//@ minicore-compile-flags: -Ccodegen-units=1

#![crate_type = "lib"]
#![feature(no_core, f16, f128)]
#![no_core]

extern crate minicore;
use minicore::*;

pub fn foo(_: f128) {}
// aarch64: !{i64 0, !"_ZTSFveE"}
// x64: !{i64 0, !"_ZTSFvgE"}
// darwin: !{i64 0, !"_ZTSFvgE"}
