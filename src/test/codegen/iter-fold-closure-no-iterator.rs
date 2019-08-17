//! Check that fold closures aren't generic in the iterator type.
// compile-flags: -C opt-level=0

fn main() {
    (0i32..10).by_ref().count();
}

// `count` calls `fold`, which calls `try_fold` -- that `fold` closure should
// not be generic in the iterator type, only in the item type.
// CHECK-NOT: {{^define.*Iterator::fold::.*closure.*Range}}
