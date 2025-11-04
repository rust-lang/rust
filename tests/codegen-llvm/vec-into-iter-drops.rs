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

/// This test case from https://users.rust-lang.org/t/unnecessary-drop-in-place-emitted-for-a-fully-consumed-intoiter/135119
/// It should not emit any `drop::<Bomb>()` because every element is forgotten.
// CHECK-LABEL: @vec_into_iter_drop_option
#[no_mangle]
pub fn vec_into_iter_drop_option(v: vec::Vec<(usize, Option<Bomb>)>) -> usize {
    // CHECK-NOT: panic
    // CHECK-NOT: Bomb$u20$as$u20$core..ops..drop..Drop
    let mut last = 0;
    v.into_iter().for_each(|(x, bomb)| {
        last = x;
        std::mem::forget(bomb);
    });
    last
}
