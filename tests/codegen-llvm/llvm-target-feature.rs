// Test that `-Zllvm-target-feature` forwards raw feature strings directly to the
// LLVM backend, bypassing Rust's known-feature table. This test uses
// `prefer-256-bit`, an LLVM-only x86 feature that is not listed in rustc's
// `target_features.rs`.

//@ add-minicore
//@ needs-llvm-components: x86
//@ compile-flags: --target x86_64-unknown-linux-gnu -Zllvm-target-feature=+prefer-256-bit
#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]
extern crate minicore;

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0

    // CHECK: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+prefer-256-bit{{.*}} }
}
