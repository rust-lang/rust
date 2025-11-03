//@ add-minicore
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Ctarget-feature=-crt-static -Zsanitizer=kcfi -Cno-prepopulate-passes -Copt-level=0

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

struct Thing;
trait MyTrait {
    // NOTE: this test assumes that this trait is dyn-compatible.
    #[unsafe(naked)]
    extern "C" fn my_naked_function(&self) {
        // the real function is defined
        // CHECK: .globl
        // CHECK-SAME: my_naked_function
        naked_asm!("ret")
    }
}
impl MyTrait for Thing {}

// the shim calls the real function
// CHECK-LABEL: define
// CHECK-SAME: my_naked_function
// CHECK-SAME: reify.shim.fnptr

// CHECK-LABEL: main
#[unsafe(no_mangle)]
pub fn main() {
    // Trick the compiler into generating an indirect call.
    const F: extern "C" fn(&Thing) = Thing::my_naked_function;

    // main calls the shim function
    // CHECK: call void
    // CHECK-SAME: my_naked_function
    // CHECK-SAME: reify.shim.fnptr
    (F)(&Thing);
}

// CHECK: declare !kcfi_type
// CHECK-SAME: my_naked_function
