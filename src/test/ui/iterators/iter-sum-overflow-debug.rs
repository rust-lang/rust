// run-pass
// ignore-wasm32-bare compiled with panic=abort by default
// compile-flags: -C debug_assertions=yes

use std::panic;

fn main() {
    let r = panic::catch_unwind(|| {
        [1, i32::max_value()].iter().sum::<i32>();
    });
    assert!(r.is_err());

    let r = panic::catch_unwind(|| {
        [2, i32::max_value()].iter().product::<i32>();
    });
    assert!(r.is_err());

    let r = panic::catch_unwind(|| {
        [1, i32::max_value()].iter().cloned().sum::<i32>();
    });
    assert!(r.is_err());

    let r = panic::catch_unwind(|| {
        [2, i32::max_value()].iter().cloned().product::<i32>();
    });
    assert!(r.is_err());
}
