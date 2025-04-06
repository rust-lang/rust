//@ add-core-stubs
//@ compile-flags: -Copt-level=3
//@revisions: with_nontemporal without_nontemporal
//@[with_nontemporal] compile-flags: --target aarch64-unknown-linux-gnu
//@[with_nontemporal] needs-llvm-components: aarch64
//@[without_nontemporal] compile-flags: --target x86_64-unknown-linux-gnu
//@[without_nontemporal] needs-llvm-components: x86

// Ensure that we *do* emit the `!nontemporal` flag on architectures where it
// is well-behaved, but do *not* emit it on architectures where it is ill-behaved.
// For more context, see <https://github.com/rust-lang/rust/issues/114582> and
// <https://github.com/llvm/llvm-project/issues/64521>.

#![feature(no_core, lang_items, intrinsics)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[rustc_intrinsic]
pub unsafe fn nontemporal_store<T>(ptr: *mut T, val: T);

#[no_mangle]
pub fn a(a: &mut u32, b: u32) {
    // CHECK-LABEL: define{{.*}}void @a
    // with_nontemporal: store i32 %b, ptr %a, align 4, !nontemporal
    // without_nontemporal-NOT: nontemporal
    unsafe {
        nontemporal_store(a, b);
    }
}
