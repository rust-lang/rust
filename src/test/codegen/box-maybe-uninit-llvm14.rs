// compile-flags: -O

// Once we're done with llvm 14 and earlier, this test can be deleted.

#![crate_type="lib"]

use std::mem::MaybeUninit;

// Boxing a `MaybeUninit` value should not copy junk from the stack
#[no_mangle]
pub fn box_uninitialized() -> Box<MaybeUninit<usize>> {
    // CHECK-LABEL: @box_uninitialized
    // CHECK-NOT: store
    // CHECK-NOT: alloca
    // CHECK-NOT: memcpy
    // CHECK-NOT: memset
    Box::new(MaybeUninit::uninit())
}

// FIXME: add a test for a bigger box. Currently broken, see
// https://github.com/rust-lang/rust/issues/58201.

// Hide the LLVM 15+ `allocalign` attribute in the declaration of __rust_alloc
// from the CHECK-NOT above. We don't check the attributes here because we can't rely
// on all of them being set until LLVM 15.
// CHECK: declare noalias{{.*}} @__rust_alloc(i{{[0-9]+}}, i{{[0-9]+.*}})
