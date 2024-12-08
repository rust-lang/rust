use std::cell::RefCell;

// Regression test for issue 7364
static boxed: Box<RefCell<isize>> = Box::new(RefCell::new(0));
//~^ ERROR `RefCell<isize>` cannot be shared between threads safely [E0277]
//~| ERROR cannot call non-const fn

fn main() { }
