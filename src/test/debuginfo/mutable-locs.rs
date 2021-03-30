// Testing the display of Cell, RefCell, and RefMut in cdb.

// cdb-only
// min-cdb-version: 10.0.18317.1001
// compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command:dx static_c,d
// cdb-check:static_c,d       [Type: core::cell::Cell<i32>]
// cdb-check:    [...] value            [Type: core::cell::UnsafeCell<i32>]

// cdb-command: dx static_c.value,d
// cdb-check:static_c.value,d [Type: core::cell::UnsafeCell<i32>]
// cdb-check:    [...] value            : 10 [Type: int]

// cdb-command:  dx dynamic_c,d
// cdb-check:dynamic_c,d      [Type: core::cell::RefCell<i32>]
// cdb-check:    [...] borrow           [Type: core::cell::Cell<isize>]
// cdb-check:    [...] value            [Type: core::cell::UnsafeCell<i32>]

// cdb-command: dx dynamic_c.value,d
// cdb-check:dynamic_c.value,d [Type: core::cell::UnsafeCell<i32>]
// cdb-check:    [...] value            : 15 [Type: int]

// cdb-command: dx b,d
// cdb-check:b,d              [Type: core::cell::RefMut<i32>]
// cdb-check:    [...] value            : [...] : 42 [Type: int *]
// cdb-check:    [...] borrow           [Type: core::cell::BorrowRefMut]

#![allow(unused_variables)]

use std::cell::{Cell, RefCell};

fn main() {
    let static_c = Cell::new(5);
    static_c.set(10);

    let dynamic_c = RefCell::new(5);
    dynamic_c.replace(15);

    let dynamic_c_0 = RefCell::new(15);
    let mut b = dynamic_c_0.borrow_mut();
    *b = 42;

    zzz(); // #break
}

fn zzz() {()}
