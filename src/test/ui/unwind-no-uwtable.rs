// run-pass
// ignore-windows target requires uwtable
// ignore-wasm32-bare no proper panic=unwind support
// compile-flags: -C panic=unwind -C force-unwind-tables=n

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
    assert!(panic::catch_unwind(AssertUnwindSafe(
        #[inline(never)]
        || increase(&mut count)
    )).is_err());
    assert_eq!(count, 1);
}
