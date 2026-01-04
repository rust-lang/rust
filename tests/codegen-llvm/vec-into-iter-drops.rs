//@ compile-flags: -Copt-level=3
// Test that we can avoid generating an element dropping loop when `vec::IntoIter` is consumed.
#![crate_type = "lib"]

use std::vec;

struct Bomb;
impl Drop for Bomb {
    #[inline]
    fn drop(&mut self) {
        panic!("dropped")
    }
}

/// Test case originally from https://users.rust-lang.org/t/unnecessary-drop-in-place-emitted-for-a-fully-consumed-intoiter/135119
///
/// What we are looking for is that there should be no calls to `impl Drop for Bomb`
/// because every element is unconditionally forgotten.
//
// CHECK-LABEL: @vec_for_each_doesnt_drop
#[no_mangle]
pub fn vec_for_each_doesnt_drop(v: vec::Vec<(usize, Option<Bomb>)>) -> usize {
    // CHECK-NOT: panic
    // CHECK-NOT: {{call.*drop_in_place}}
    // CHECK-NOT: Bomb$u20$as$u20$core..ops..drop..Drop
    let mut last = 0;
    v.into_iter().for_each(|(x, bomb)| {
        last = x;
        std::mem::forget(bomb);
    });
    last
}

/// Test that does *not* get the above optimization we are expecting:
/// it uses a normal `for` loop which calls `Iterator::next()` and then drops the iterator,
/// and dropping the iterator drops remaining items.
///
/// This test exists to prove that the above CHECK-NOT is looking for the right things.
/// However, it might start failing if LLVM figures out that there are no remaining items.
//
// CHECK-LABEL: @vec_for_loop
#[no_mangle]
pub fn vec_for_loop(v: vec::Vec<(usize, Option<Bomb>)>) -> usize {
    // CHECK: {{call.*drop_in_place}}
    let mut last = 0;
    for (x, bomb) in v {
        last = x;
        std::mem::forget(bomb);
    }
    last
}

/// Test where there still should be drops because there are no forgets.
///
/// This test exists to prove that the above CHECK-NOT is looking for the right things
/// and does not say anything interesting about codegen itself.
//
// CHECK-LABEL: @vec_for_each_does_drop
#[no_mangle]
pub fn vec_for_each_does_drop(v: vec::Vec<(usize, Option<Bomb>)>) -> usize {
    // CHECK: begin_panic
    let mut last = 0;
    v.into_iter().for_each(|(x, bomb)| {
        last = x;
    });
    last
}
