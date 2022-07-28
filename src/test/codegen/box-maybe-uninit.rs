// compile-flags: -O
// min-llvm-version: 15.0
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

// Hide the `allocalign` attribute in the declaration of __rust_alloc
// from the CHECK-NOT above, and also verify the attributes got set reasonably.
// CHECK: declare noalias ptr @__rust_alloc(i{{[0-9]+}}, i{{[0-9]+}} allocalign) unnamed_addr [[RUST_ALLOC_ATTRS:#[0-9]+]]

// CHECK-DAG: attributes [[RUST_ALLOC_ATTRS]] = { {{.*}} allockind("alloc,uninitialized,aligned") allocsize(0) uwtable "alloc-family"="__rust_alloc" {{.*}} }
