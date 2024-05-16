// compile-flags: --crate-type=rlib -Copt-level=0
// revisions: force-on aarch64-apple aarch64-apple-off
// [force-on] compile-flags: -Cforce-frame-pointers=on
// [aarch64-apple] needs-llvm-components: aarch64
// [aarch64-apple] compile-flags: --target=aarch64-apple-darwin
// [aarch64-apple-off] needs-llvm-components: aarch64
// [aarch64-apple-off] compile-flags: --target=aarch64-apple-darwin -Cforce-frame-pointers=off
/*
Tests that the frame pointers can be controlled by the CLI. We find aarch64-apple-darwin useful
because of its icy-clear policy regarding frame pointers (software SHALL be compiled with them),
e.g. https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms says:

* The frame pointer register (x29) must always address a valid frame record. Some functions —
  such as leaf functions or tail calls — may opt not to create an entry in this list.
  As a result, stack traces are always meaningful, even without debug information.
*/
#![feature(no_core, lang_items)]
#![no_core]
#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}
impl Copy for u32 {}

// CHECK: define i32 @peach{{.*}}[[PEACH_ATTRS:\#[0-9]+]] {
#[no_mangle]
pub fn peach(x: u32) -> u32 {
    x
}

// CHECK: attributes [[PEACH_ATTRS]] = {
// force-on-SAME: {{.*}}"frame-pointer"="all"
// aarch64-apple-SAME: {{.*}}"frame-pointer"="all"
// aarch64-apple-off-NOT: {{.*}}"frame-pointer"{{.*}}
// CHECK-SAME: }
