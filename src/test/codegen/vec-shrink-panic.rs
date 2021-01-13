// compile-flags: -O
// ignore-debug: the debug assertions get in the way
#![crate_type = "lib"]
#![feature(shrink_to)]

// Make sure that `Vec::shrink_to_fit` never emits panics via `RawVec::shrink_to_fit`,
// "Tried to shrink to a larger capacity", because the length is *always* <= capacity.

// CHECK-LABEL: @shrink_to_fit
#[no_mangle]
pub fn shrink_to_fit(vec: &mut Vec<u32>) {
    // CHECK-NOT: panic
    vec.shrink_to_fit();
}

// CHECK-LABEL: @issue71861
#[no_mangle]
pub fn issue71861(vec: Vec<u32>) -> Box<[u32]> {
    // CHECK-NOT: panic
    vec.into_boxed_slice()
}

// CHECK-LABEL: @issue75636
#[no_mangle]
pub fn issue75636<'a>(iter: &[&'a str]) -> Box<[&'a str]> {
    // CHECK-NOT: panic
    iter.iter().copied().collect()
}

// Sanity-check that we do see a possible panic for an arbitrary `Vec::shrink_to`.
// CHECK-LABEL: @shrink_to
#[no_mangle]
pub fn shrink_to(vec: &mut Vec<u32>) {
    // CHECK: panic
    vec.shrink_to(42);
}
