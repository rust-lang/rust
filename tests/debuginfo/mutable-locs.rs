// Testing the display of Cell, RefCell, and RefMut in cdb.

// cdb-only
//@ min-cdb-version: 10.0.18317.1001
//@ compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command:dx static_c,d
// cdb-check:static_c,d       : 10 [Type: core::cell::Cell<i32>]
// cdb-check:    [<Raw View>]     [Type: core::cell::Cell<i32>]

// cdb-command: dx static_c.value,d
// cdb-check:static_c.value,d : 10 [Type: core::cell::UnsafeCell<i32>]
// cdb-check:    [<Raw View>]     [Type: core::cell::UnsafeCell<i32>]

// cdb-command:  dx dynamic_c,d
// cdb-check:dynamic_c,d      : 15 [Type: core::cell::RefCell<i32>]
// cdb-check:    [<Raw View>]     [Type: core::cell::RefCell<i32>]
// cdb-check:    [Borrow state]   : Unborrowed

// cdb-command: dx dynamic_c.value,d
// cdb-check:dynamic_c.value,d : 15 [Type: core::cell::UnsafeCell<i32>]
// cdb-check:    [<Raw View>]     [Type: core::cell::UnsafeCell<i32>]

// cdb-command: dx b,d
// cdb-check:b,d              : 42 [Type: core::cell::RefMut<i32>]
// cdb-check:    [<Raw View>]     [Type: core::cell::RefMut<i32>]
// cdb-check:    42 [Type: int]

// cdb-command: g

// cdb-command: dx dynamic_c,d
// cdb-check:dynamic_c,d      : 15 [Type: core::cell::RefCell<i32>]
// cdb-check:    [<Raw View>]     [Type: core::cell::RefCell<i32>]
// cdb-check:    [Borrow state]   : Immutably borrowed

// cdb-command: dx r_borrow,d
// cdb-check:r_borrow,d       : 15 [Type: core::cell::Ref<i32>]
// cdb-check:    [<Raw View>]     [Type: core::cell::Ref<i32>]
// cdb-check:    15 [Type: int]

// cdb-command: g

// cdb-command: dx dynamic_c,d
// cdb-check:dynamic_c,d      : 15 [Type: core::cell::RefCell<i32>]
// cdb-check:    [<Raw View>]     [Type: core::cell::RefCell<i32>]
// cdb-check:    [Borrow state]   : Unborrowed

// cdb-command: g

// cdb-command: dx dynamic_c,d
// cdb-check:dynamic_c,d      : 15 [Type: core::cell::RefCell<i32>]
// cdb-check:    [<Raw View>]     [Type: core::cell::RefCell<i32>]
// cdb-check:    [Borrow state]   : Mutably borrowed

// cdb-command: dx r_borrow_mut,d
// cdb-check:r_borrow_mut,d   : 15 [Type: core::cell::RefMut<i32>]
// cdb-check:    [<Raw View>]     [Type: core::cell::RefMut<i32>]
// cdb-check:    15 [Type: int]

// cdb-command: g

// cdb-command: dx dynamic_c,d
// cdb-check:dynamic_c,d      : 15 [Type: core::cell::RefCell<i32>]
// cdb-check:    [<Raw View>]     [Type: core::cell::RefCell<i32>]
// cdb-check:    [Borrow state]   : Unborrowed

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

    // Check that `RefCell`'s borrow state visualizes correctly
    {
        let r_borrow = dynamic_c.borrow();
        zzz(); // #break
    }

    zzz(); // #break

    {
        let r_borrow_mut = dynamic_c.borrow_mut();
        zzz(); // #break
    }

    zzz(); // #break
}

fn zzz() {()}
