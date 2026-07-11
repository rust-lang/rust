//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

// Regression test for https://github.com/rust-lang/rust/issues/159116
// where a change had started reading at the wrong offset into a const
// when the sub-object had BackendRepr::ScalarPair

const PAIRS: [(u64, u64); 3] = [(1, 2), (3, 4), (5, 6)];

// CHECK-LABEL: @read_not_first_pair
#[no_mangle]
pub fn read_not_first_pair() -> (u64, u64) {
    // FIXME: This is clearly wrong

    // CHECK: start:
    // CHECK-NEXT: ret { i64, i64 } { i64 3, i64 2 }
    PAIRS[1]
}
