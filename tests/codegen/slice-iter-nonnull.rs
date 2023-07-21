// no-system-llvm
// compile-flags: -O
// ignore-debug (these add extra checks that make it hard to verify)
#![crate_type = "lib"]
#![feature(exact_size_is_empty)]

// The slice iterator used to `assume` that the `start` pointer was non-null.
// That ought to be unneeded, though, since the type is `NonNull`, so this test
// confirms that the appropriate metadata is included to denote that.

// It also used to `assume` the `end` pointer was non-null, but that's no longer
// needed as the code changed to read it as a `NonNull`, and thus gets the
// appropriate `!nonnull` annotations naturally.

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

// The slice iterator `new` methods used to `assume` that the pointer is non-null,
// but passing slices already requires that, to the extent that LLVM actually
// removed the `call @llvm.assume` anyway.  These tests just demonstrate that the
// attribute is there, and confirms adding the assume back doesn't do anything.

// CHECK-LABEL: @slice_iter_new
// CHECK-SAME: (ptr noalias noundef nonnull {{.+}} %slice.0, {{.+}} noundef %slice.1)
#[no_mangle]
pub fn slice_iter_new(slice: &[u32]) -> std::slice::Iter<'_, u32> {
    // CHECK-NOT: slice
    // CHECK: %[[END:.+]] = getelementptr inbounds i32{{.+}} %slice.0{{.+}} %slice.1
    // CHECK-NOT: slice
    // CHECK: insertvalue {{.+}} ptr %slice.0, 0
    // CHECK-NOT: slice
    // CHECK: insertvalue {{.+}} ptr %[[END]], 1
    // CHECK-NOT: slice
    // CHECK: }
    slice.iter()
}

// CHECK-LABEL: @slice_iter_mut_new
// CHECK-SAME: (ptr noalias noundef nonnull {{.+}} %slice.0, {{.+}} noundef %slice.1)
#[no_mangle]
pub fn slice_iter_mut_new(slice: &mut [u32]) -> std::slice::IterMut<'_, u32> {
    // CHECK-NOT: slice
    // CHECK: %[[END:.+]] = getelementptr inbounds i32{{.+}} %slice.0{{.+}} %slice.1
    // CHECK-NOT: slice
    // CHECK: insertvalue {{.+}} ptr %slice.0, 0
    // CHECK-NOT: slice
    // CHECK: insertvalue {{.+}} ptr %[[END]], 1
    // CHECK-NOT: slice
    // CHECK: }
    slice.iter_mut()
}

// CHECK-LABEL: @slice_iter_is_empty
#[no_mangle]
pub fn slice_iter_is_empty(it: &std::slice::Iter<'_, u32>) -> bool {
    // CHECK: %[[ENDP:.+]] = getelementptr{{.+}}ptr %it,{{.+}} 1
    // CHECK: %[[END:.+]] = load ptr, ptr %[[ENDP]]
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: %[[START:.+]] = load ptr, ptr %it,
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef

    // CHECK: %[[RET:.+]] = icmp eq ptr %[[START]], %[[END]]
    // CHECK: ret i1 %[[RET]]
    it.is_empty()
}

// CHECK-LABEL: @slice_iter_len
#[no_mangle]
pub fn slice_iter_len(it: &std::slice::Iter<'_, u32>) -> usize {
    // CHECK: %[[START:.+]] = load ptr, ptr %it,
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: %[[ENDP:.+]] = getelementptr{{.+}}ptr %it,{{.+}} 1
    // CHECK: %[[END:.+]] = load ptr, ptr %[[ENDP]]
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef

    // CHECK: ptrtoint
    // CHECK: ptrtoint
    // CHECK: sub nuw
    // CHECK: lshr exact
    it.len()
}
