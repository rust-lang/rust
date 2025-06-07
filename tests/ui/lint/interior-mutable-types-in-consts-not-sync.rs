//@ check-pass

use std::cell::{Cell, RefCell, UnsafeCell, LazyCell, OnceCell};

const A: Cell<i32> = Cell::new(0);
//~^ WARN interior mutability in `const` item
const B: RefCell<i32> = RefCell::new(0);
//~^ WARN interior mutability in `const` item
const C: UnsafeCell<i32> = UnsafeCell::new(0);
//~^ WARN interior mutability in `const` item
const D: LazyCell<i32> = LazyCell::new(|| 0);
//~^ WARN interior mutability in `const` item
const E: OnceCell<i32> = OnceCell::new();
//~^ WARN interior mutability in `const` item

fn main() {}
