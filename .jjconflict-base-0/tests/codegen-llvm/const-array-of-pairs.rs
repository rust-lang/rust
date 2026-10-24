//@ compile-flags: -O -C no-prepopulate-passes

#![crate_type = "lib"]

// Regression test for https://github.com/rust-lang/rust/issues/159116
// where a change had started reading at the wrong offset into a const
// when the sub-object had BackendRepr::ScalarPair

const PAIRS: [(u16, u16); 3] = [(1, 2), (3, 4), (5, 6)];

// CHECK-LABEL: @read_not_first_pair
#[no_mangle]
pub fn read_not_first_pair() -> (u16, u16) {
    // CHECK: start:
    // CHECK-NEXT: ret { i16, i16 } { i16 3, i16 4 }
    PAIRS[1]
}
