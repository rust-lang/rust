//! Check that fold closures aren't duplicated for each iterator type.
// compile-flags: -C opt-level=0

fn main() {
    (0i32..10).by_ref().count();
    (0i32..=10).by_ref().count();
}

// `count` calls `fold`, which calls `try_fold` -- find the `fold` closure:
// CHECK: {{^define.*Iterator::fold::.*closure}}
//
// Only one closure is needed for both `count` calls, even from different
// monomorphized iterator types, as it's only generic over the item type.
// CHECK-NOT: {{^define.*Iterator::fold::.*closure}}
