// no-system-llvm
// compile-flags: -O
// ignore-debug (these add extra checks that make it hard to verify)
#![crate_type = "lib"]

// The slice iterator used to `assume` that the `start` pointer was non-null.
// That ought to be unneeded, though, since the type is `NonNull`, so this test
// confirms that the appropriate metadata is included to denote that.

// CHECK-LABEL: @slice_iter_next(
#[no_mangle]
pub fn slice_iter_next<'a>(it: &mut std::slice::Iter<'a, u32>) -> Option<&'a u32> {
    // CHECK: %[[ENDP:.+]] = getelementptr{{.+}}ptr %it,{{.+}} 1
    // CHECK: %[[END:.+]] = load ptr, ptr %[[ENDP]]
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: %[[START:.+]] = load ptr, ptr %it,
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: icmp eq ptr %[[START]], %[[END]]

    // CHECK: store ptr{{.+}}, ptr %it,

    it.next()
}

// CHECK-LABEL: @slice_iter_next_back(
#[no_mangle]
pub fn slice_iter_next_back<'a>(it: &mut std::slice::Iter<'a, u32>) -> Option<&'a u32> {
    // CHECK: %[[ENDP:.+]] = getelementptr{{.+}}ptr %it,{{.+}} 1
    // CHECK: %[[END:.+]] = load ptr, ptr %[[ENDP]]
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: %[[START:.+]] = load ptr, ptr %it,
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: icmp eq ptr %[[START]], %[[END]]

    // CHECK: store ptr{{.+}}, ptr %[[ENDP]],

    it.next_back()
}
