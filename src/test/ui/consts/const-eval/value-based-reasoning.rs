// Things that we *can* take references of in consts even though their
// type is not Freeze+Sync.

// compile-pass

use std::cell::Cell;

// test const qualification
const CLOSURE: &Fn() = &|| {};
const NO_CELL: &Option<Cell<i32>> = &None;

// test promotion
fn mk_closure() -> &'static Fn() { &|| {} }
fn mk_no_cell() -> &'static Option<Cell<i32>> { &None }

fn main() {}
