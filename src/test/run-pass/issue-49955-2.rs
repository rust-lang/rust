// compile-flags: -Z borrowck=mir

use std::cell::Cell;

#[inline(never)]
fn tuple_field() -> &'static u32 {
    // This test is MIR-borrowck-only because the old borrowck
    // doesn't agree that borrows of "frozen" (i.e. without any
    // interior mutability) fields of non-frozen temporaries,
    // should be promoted, while MIR promotion does promote them.
    &(Cell::new(5), 42).1
}

fn main() {
    assert_eq!(tuple_field().to_string(), "42");
}
