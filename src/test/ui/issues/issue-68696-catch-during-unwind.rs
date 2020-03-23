// Checks that catch_unwind can be used if unwinding is already in progress.
// Used to fail when standard library had been compiled with debug assertions,
// due to incorrect assumption that a current thread is not panicking when
// entering the catch_unwind.
//
// run-pass
// ignore-wasm       no panic support
// ignore-emscripten no panic support

use std::panic::catch_unwind;

#[derive(Default)]
struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = catch_unwind(|| {});
    }
}

fn main() {
    let _ = catch_unwind(|| {
        let _guard = Guard::default();
        panic!();
    });
}
