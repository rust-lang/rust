//@ compile-flags: -O
// Regression test for https://github.com/rust-lang/rust/issues/149480:
// the bounds check should be eliminated when indexing an array with
// the result of an exhaustive match over nested enums. The range
// assume emitted by MatchBranchSimplification after the IntToInt cast
// allows LLVM to prove the index is in-bounds.

#![crate_type = "lib"]

pub enum Foo {
    A(A),
    B(B),
}
pub enum A {
    A0,
    A1,
    A2,
}
pub enum B {
    B0,
    B1,
}

// CHECK-LABEL: @bar
#[no_mangle]
pub fn bar(foo: Foo, arr: &[u8; 5]) -> u8 {
    let offset: usize = match foo {
        Foo::A(A::A0) => 0,
        Foo::A(A::A1) => 1,
        Foo::A(A::A2) => 2,
        Foo::B(B::B0) => 3,
        Foo::B(B::B1) => 4,
    };
    // The bounds check must be eliminated.
    // CHECK-NOT: panic_bounds_check
    // Positive check: the indexing must lower to a plain load from `arr`,
    // so the test cannot pass accidentally if `bar` is optimized into
    // another kind of panicking path or if `panic_bounds_check` is
    // renamed.
    // CHECK:     load i8, ptr
    // CHECK:     ret i8
    arr[offset]
}

// Sanity check: make sure `panic_bounds_check` is still the symbol LLVM
// emits for a non-elidable out-of-bounds index, so the `CHECK-NOT` above
// is guarding against something real.
// CHECK-LABEL: @test_check
#[no_mangle]
pub fn test_check(arr: &[u8], i: usize) -> u8 {
    // CHECK: panic_bounds_check
    arr[i]
}
