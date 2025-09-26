// Checks that catch_unwind can be used if unwinding is already in progress.
// Used to fail when standard library had been compiled with debug assertions,
// due to incorrect assumption that a current thread is not panicking when
// entering the catch_unwind.
//
//@ run-pass
//@ ignore-backends: gcc

use std::panic::catch_unwind;

#[allow(dead_code)]
#[derive(Default)]
struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = catch_unwind(|| {});
    }
}

fn main() {
    #[cfg(panic = "unwind")]
    let _ = catch_unwind(|| {
        let _guard = Guard::default();
        panic!();
    });
}
