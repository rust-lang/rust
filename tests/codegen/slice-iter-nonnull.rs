//@ compile-flags: -Copt-level=3
//@ needs-deterministic-layouts
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
    // CHECK: %[[START:.+]] = load ptr, ptr %it,
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: %[[ENDP:.+]] = getelementptr inbounds{{( nuw)?}} i8, ptr %it, {{i32 4|i64 8}}
    // CHECK: %[[END:.+]] = load ptr, ptr %[[ENDP]]
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: icmp eq ptr %[[START]], %[[END]]

    // CHECK-NOT: store
    // CHECK: store ptr {{.+}}, ptr %it,
    // CHECK-NOT: store

    it.next()
}

// CHECK-LABEL: @slice_iter_next_back(
#[no_mangle]
pub fn slice_iter_next_back<'a>(it: &mut std::slice::Iter<'a, u32>) -> Option<&'a u32> {
    // CHECK: %[[START:.+]] = load ptr, ptr %it,
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: %[[ENDP:.+]] = getelementptr inbounds{{( nuw)?}} i8, ptr %it, {{i32 4|i64 8}}
    // CHECK: %[[END:.+]] = load ptr, ptr %[[ENDP]]
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: icmp eq ptr %[[START]], %[[END]]

    // CHECK-NOT: store
    // CHECK: store {{i32|i64}} {{.+}}, ptr %[[ENDP]],
    // CHECK-NOT: store

    it.next_back()
}

// The slice iterator `new` methods used to `assume` that the pointer is non-null,
// but passing slices already requires that, to the extent that LLVM actually
// removed the `call @llvm.assume` anyway.  These tests just demonstrate that the
// attribute is there, and confirms adding the assume back doesn't do anything.

// CHECK-LABEL: @slice_iter_new
// CHECK-SAME: (ptr noalias noundef nonnull {{.+}} %slice.0, [[USIZE:i32|i64]] noundef %slice.1)
#[no_mangle]
pub fn slice_iter_new(slice: &[u32]) -> std::slice::Iter<'_, u32> {
    // CHECK-NOT: slice
    // CHECK: %[[END_PTR:.+]] = getelementptr inbounds{{( nuw)?}} i32, ptr %slice.0, [[USIZE]] %slice.1
    // CHECK-NEXT: %[[END_ADDR:.+]] = ptrtoint ptr %[[END_PTR]] to [[USIZE]]
    // CHECK-NOT: slice
    // CHECK: insertvalue {{.+}} ptr %slice.0, 0
    // CHECK-NOT: slice
    // CHECK: insertvalue {{.+}} [[USIZE]] %[[END_ADDR]], 1
    // CHECK-NOT: slice
    // CHECK: }
    slice.iter()
}

// CHECK-LABEL: @slice_iter_mut_new
// CHECK-SAME: (ptr noalias noundef nonnull {{.+}} %slice.0, [[USIZE:i32|i64]] noundef %slice.1)
#[no_mangle]
pub fn slice_iter_mut_new(slice: &mut [u32]) -> std::slice::IterMut<'_, u32> {
    // CHECK-NOT: slice
    // CHECK: %[[END_PTR:.+]] = getelementptr inbounds{{( nuw)?}} i32, ptr %slice.0, [[USIZE]] %slice.1
    // CHECK-NEXT: %[[END_ADDR:.+]] = ptrtoint ptr %[[END_PTR]] to [[USIZE]]
    // CHECK-NOT: slice
    // CHECK: insertvalue {{.+}} ptr %slice.0, 0
    // CHECK-NOT: slice
    // CHECK: insertvalue {{.+}} [[USIZE]] %[[END_ADDR]], 1
    // CHECK-NOT: slice
    // CHECK: }
    slice.iter_mut()
}

// CHECK-LABEL: @slice_iter_is_empty
#[no_mangle]
pub fn slice_iter_is_empty(it: &std::slice::Iter<'_, u32>) -> bool {
    // CHECK: %[[START:.+]] = load ptr, ptr %it,
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: %[[ENDP:.+]] = getelementptr inbounds{{( nuw)?}} i8, ptr %it, {{i32 4|i64 8}}
    // CHECK: %[[END:.+]] = load ptr, ptr %[[ENDP]]
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
    // CHECK: %[[ENDP:.+]] = getelementptr inbounds{{( nuw)?}} i8, ptr %it, {{i32 4|i64 8}}
    // CHECK: %[[END:.+]] = load ptr, ptr %[[ENDP]]
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef

    // CHECK: %[[END_ADDR:.+]] = ptrtoint ptr %[[END]]
    // CHECK: %[[START_ADDR:.+]] = ptrtoint ptr %[[START]]
    // CHECK: %[[BYTES:.+]] = sub nuw [[USIZE]] %[[END_ADDR]], %[[START_ADDR]]
    // CHECK: %[[ELEMS:.+]] = lshr exact [[USIZE]] %[[BYTES]], 2
    // CHECK: ret [[USIZE]] %[[ELEMS]]
    it.len()
}
