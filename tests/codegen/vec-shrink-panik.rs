// revisions: old new
// LLVM 17 realizes double panic is not possible and doesn't generate calls
// to panic_cannot_unwind.
// [old]ignore-llvm-version: 17 - 99
// [new]min-llvm-version: 17
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

    // Call to panic_cannot_unwind in case of double-panic is expected
    // on LLVM 16 and older, but other panics are not.
    // CHECK: cleanup
    // old-NEXT: ; call core::panicking::panic_cannot_unwind
    // old-NEXT: panic_cannot_unwind

    // CHECK-NOT: panic
    vec.into_boxed_slice()
}

// CHECK-LABEL: @issue75636
#[no_mangle]
pub fn issue75636<'a>(iter: &[&'a str]) -> Box<[&'a str]> {
    // CHECK-NOT: panic

    // Call to panic_cannot_unwind in case of double-panic is expected,
    // on LLVM 16 and older, but other panics are not.
    // CHECK: cleanup
    // old-NEXT: ; call core::panicking::panic_cannot_unwind
    // old-NEXT: panic_cannot_unwind

    // CHECK-NOT: panic
    iter.iter().copied().collect()
}

// old: ; core::panicking::panic_cannot_unwind
// old: declare void @{{.*}}panic_cannot_unwind
