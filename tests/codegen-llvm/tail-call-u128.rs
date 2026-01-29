//@ add-minicore
//@ revisions: win linux
//
//@ compile-flags: -Copt-level=3
//@[linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[linux] needs-llvm-components: x86
//@[win] compile-flags: --target x86_64-pc-windows-msvc
//@[win] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core, lang_items, explicit_tail_calls)]
#![allow(incomplete_features)]
#![no_core]

extern crate minicore;
use minicore::*;

// linux: define noundef i128 @foo(i128 noundef %a, i128 noundef %b)
// win: define <16 x i8> @foo(ptr {{.*}} %a, ptr {{.*}} %b)
#[unsafe(no_mangle)]
#[inline(never)]
extern "C" fn foo(a: u128, b: u128) -> u128 {
    // linux: start:
    // linux-NEXT: musttail call noundef i128 @bar(i128 noundef %b, i128 noundef %a)
    //
    //
    // win: start:
    // win-NEXT: %0 = load i128, ptr %b
    // win-NEXT: %1 = load i128, ptr %a
    // win-NEXT: store i128 %0, ptr %a
    // win-NEXT: store i128 %1, ptr %b
    // win-NEXT: musttail call <16 x i8> @bar(ptr {{.*}} %a, ptr {{.*}} %b)
    become bar(b, a)
}

unsafe extern "C" {
    safe fn bar(a: u128, b: u128) -> u128;
}
