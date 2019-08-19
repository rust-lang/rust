// run-pass
// Test use of stabilized const fns in std formerly using individual feature gates.

use std::cell::Cell;

const CELL: Cell<i32> = Cell::new(42);

fn main() {
    let v = CELL.get();
    CELL.set(v+1);

    assert_eq!(CELL.get(), v);
}
