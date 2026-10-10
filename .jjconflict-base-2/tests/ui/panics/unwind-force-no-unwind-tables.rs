//! This test checks that Rust's unwinding mechanism correctly executes `Drop`
//! implementations during stack unwinding, even when unwind tables (`uwtable`)
//! are explicitly disabled via `-C force-unwind-tables=n`.

//@ run-pass
//@ needs-unwind
//@ ignore-windows target requires uwtable
//@ compile-flags: -C panic=unwind -C force-unwind-tables=n
//@ ignore-backends: gcc

use std::panic::{self, AssertUnwindSafe};

struct Increase<'a>(&'a mut u8);

impl Drop for Increase<'_> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

#[inline(never)]
fn unwind() {
    panic!();
}

#[inline(never)]
fn increase(count: &mut u8) {
    let _increase = Increase(count);
    unwind();
}

fn main() {
    let mut count = 0;
    assert!(
        panic::catch_unwind(AssertUnwindSafe(
            #[inline(never)]
            || increase(&mut count)
        ))
        .is_err()
    );
    assert_eq!(count, 1);
}
