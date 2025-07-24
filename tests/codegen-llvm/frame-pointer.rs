//@ add-core-stubs
//@ compile-flags: --crate-type=rlib -Copt-level=0
//@ revisions: aarch64-apple aarch64-linux force x64-apple x64-linux
//@ [aarch64-apple] needs-llvm-components: aarch64
//@ [aarch64-apple] compile-flags: --target=aarch64-apple-darwin
//@ [aarch64-linux] needs-llvm-components: aarch64
//@ [aarch64-linux] compile-flags: --target=aarch64-unknown-linux-gnu
//@ [force] needs-llvm-components: x86
//@ [force] compile-flags: --target=x86_64-unknown-linux-gnu -Cforce-frame-pointers=yes
//@ [x64-apple] needs-llvm-components: x86
//@ [x64-apple] compile-flags: --target=x86_64-apple-darwin
//@ [x64-linux] needs-llvm-components: x86
//@ [x64-linux] compile-flags: --target=x86_64-unknown-linux-gnu

#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK: define i32 @peach{{.*}}[[PEACH_ATTRS:\#[0-9]+]] {
#[no_mangle]
pub fn peach(x: u32) -> u32 {
    x
}

// CHECK: attributes [[PEACH_ATTRS]] = {
// x64-linux-NOT: {{.*}}"frame-pointer"{{.*}}
// x64-apple-SAME: {{.*}}"frame-pointer"="all"
// force-SAME: {{.*}}"frame-pointer"="all"
//
// AAPCS64 demands frame pointers:
// aarch64-linux-SAME: {{.*}}"frame-pointer"="non-leaf"
// aarch64-apple-SAME: {{.*}}"frame-pointer"="non-leaf"
// CHECK-SAME: }
