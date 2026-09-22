// Regression test for https://github.com/rust-lang/rust/issues/161375
// Checking that an empty BTree's drop is optimized away.
//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::collections::BTreeMap;

// CHECK-LABEL: @drop_btree
// CHECK-NOT: dying_next
// CHECK: ret void
#[no_mangle]
pub fn drop_btree() {
    let _ = BTreeMap::<(), ()>::new();
}
