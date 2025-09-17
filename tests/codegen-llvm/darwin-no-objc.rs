// Test that we don't generate Objective-C definitions or image info unnecessarily.

//@ add-core-stubs
//@ revisions: i686_apple_darwin
//@ [i686_apple_darwin] compile-flags: --target i686-apple-darwin
//@ [i686_apple_darwin] needs-llvm-components: x86
//@ revisions: x86_64_macos
//@ [x86_64_macos] compile-flags: --target x86_64-apple-darwin
//@ [x86_64_macos] needs-llvm-components: x86
//@ revisions: aarch64_macos
//@ [aarch64_macos] compile-flags: --target aarch64-apple-darwin
//@ [aarch64_macos] needs-llvm-components: aarch64
//@ revisions: i386_ios
//@ [i386_ios] compile-flags: --target i386-apple-ios
//@ [i386_ios] needs-llvm-components: x86
//@ revisions: x86_64_ios
//@ [x86_64_ios] compile-flags: --target x86_64-apple-ios
//@ [x86_64_ios] needs-llvm-components: x86
//@ revisions: armv7s_ios
//@ [armv7s_ios] compile-flags: --target armv7s-apple-ios
//@ [armv7s_ios] needs-llvm-components: arm
//@ revisions: aarch64_ios
//@ [aarch64_ios] compile-flags: --target aarch64-apple-ios
//@ [aarch64_ios] needs-llvm-components: aarch64
//@ revisions: aarch64_ios_sim
//@ [aarch64_ios_sim] compile-flags: --target aarch64-apple-ios-sim
//@ [aarch64_ios_sim] needs-llvm-components: aarch64

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn foo() {}

// CHECK-NOT: %struct._class_t
// CHECK-NOT: %struct._objc_module
// CHECK-NOT: @OBJC_CLASS_NAME_
// CHECK-NOT: @"OBJC_CLASS_$_{{[0-9A-Z_a-z]+}}"
// CHECK-NOT: @"OBJC_CLASSLIST_REFERENCES_$_.{{[0-9]+}}"
// CHECK-NOT: @OBJC_METH_VAR_NAME_
// CHECK-NOT: @OBJC_SELECTOR_REFERENCES_
// CHECK-NOT: @OBJC_MODULES

// CHECK-NOT: !"Objective-C Version"
// CHECK-NOT: !"Objective-C Image Info Version"
// CHECK-NOT: !"Objective-C Image Info Section"
// CHECK-NOT: !"Objective-C Is Simulated"
// CHECK-NOT: !"Objective-C Class Properties"
