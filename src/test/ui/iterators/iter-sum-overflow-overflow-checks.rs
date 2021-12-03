// run-pass
// needs-unwind
// ignore-wasm32-bare compiled with panic=abort by default
// compile-flags: -C overflow-checks

use std::panic;

fn main() {
    let r = panic::catch_unwind(|| {
        [1, i32::MAX].iter().sum::<i32>();
    });
    assert!(r.is_err());

    let r = panic::catch_unwind(|| {
        [2, i32::MAX].iter().product::<i32>();
    });
    assert!(r.is_err());

    let r = panic::catch_unwind(|| {
        [1, i32::MAX].iter().cloned().sum::<i32>();
    });
    assert!(r.is_err());

    let r = panic::catch_unwind(|| {
        [2, i32::MAX].iter().cloned().product::<i32>();
    });
    assert!(r.is_err());
}
