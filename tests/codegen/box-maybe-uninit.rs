// compile-flags: -O
// min-llvm-version: 15.0
#![crate_type = "lib"]

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

// https://github.com/rust-lang/rust/issues/58201
#[no_mangle]
pub fn box_uninitialized2() -> Box<MaybeUninit<[usize; 1024 * 1024]>> {
    // CHECK-LABEL: @box_uninitialized2
    // CHECK-NOT: store
    // CHECK-NOT: alloca
    // CHECK-NOT: memcpy
    // CHECK-NOT: memset
    Box::new(MaybeUninit::uninit())
}

// Hide the `allocalign` attribute in the declaration of __rust_alloc
// from the CHECK-NOT above, and also verify the attributes got set reasonably.
// CHECK: declare {{(dso_local )?}}noalias noundef ptr @__rust_alloc(i{{[0-9]+}} noundef, i{{[0-9]+}} allocalign noundef) unnamed_addr [[RUST_ALLOC_ATTRS:#[0-9]+]]

// CHECK-DAG: attributes [[RUST_ALLOC_ATTRS]] = { {{.*}} allockind("alloc,uninitialized,aligned") allocsize(0) {{(uwtable )?}}"alloc-family"="__rust_alloc" {{.*}} }
