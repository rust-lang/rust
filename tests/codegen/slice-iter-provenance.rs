//@ compile-flags: -O
#![crate_type = "lib"]

// Back when `slice::Iter(Mut)` carried two pointers, LLVM was not *allowed*
// to optimize out certain things at the IR level since starting from the other
// end had a difference provenance and thus wasn't actually equivalent.

// Now that they're `{ ptr, usize }`, however, there's only one provenance used
// by everything producing pointers

// FIXME: Add more tests here LLVM has fixed the following bug:
// <https://github.com/llvm/llvm-project/issues/86417>

// CHECK-LABEL: @first_via_nth_back
#[no_mangle]
pub unsafe fn first_via_nth_back(mut it: std::slice::Iter<'_, i8>) -> &i8 {
    // CHECK: ret ptr %0
    let len = it.len();
    it.nth_back(len - 1).unwrap_unchecked()
}
