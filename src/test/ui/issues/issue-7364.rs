#![feature(box_syntax)]

use std::cell::RefCell;

// Regression test for issue 7364
static boxed: Box<RefCell<isize>> = box RefCell::new(0);
//~^ ERROR allocations are not allowed in statics
//~| ERROR `std::cell::RefCell<isize>` cannot be shared between threads safely [E0277]
//~| ERROR static contains unimplemented expression type

fn main() { }
