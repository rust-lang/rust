// Regression test for #66630.
use std::sync::Mutex;
use std::cell::RefCell;
use std::borrow::Borrow;

struct A {
    child: Mutex<RefCell<i32>>,
}

impl A {
    fn is_good(&self) -> bool {
        let cell = self.child.lock().unwrap();
        let opt_target = cell.borrow();
        //~^ NOTE: this method is `std::borrow::Borrow::borrow`
        //~| NOTE: this method is `std::borrow::Borrow::borrow`
        let _ = opt_target.abs(); //~ ERROR: no method named `abs` found for
        //~^ NOTE: method not found in `&std::sync::MutexGuard<'_, RefCell<i32>>`
        //~| NOTE: this returns `&std::sync::MutexGuard<'_, RefCell<i32>>`
        //~| NOTE: you might have meant to call `std::cell::RefCell::<T>::borrow` instead
        let _ = *opt_target > 1; //~ ERROR: binary operation `>` cannot be applied
        //~^ NOTE: std::sync::MutexGuard<'_, RefCell<i32>>
        //~| NOTE: `std::sync::MutexGuard<'_, RefCell<i32>>` does not implement `PartialOrd<{integer}>`
        //~| NOTE: this returns `std::sync::MutexGuard<'_, RefCell<i32>>`
        //~| NOTE: you might have meant to call `std::cell::RefCell::<T>::borrow` instead
        //~| NOTE: {integer}
        1 < *opt_target //~ ERROR: E0277
        //~^ NOTE: no implementation for `{integer} < std::sync::MutexGuard<'_, RefCell<i32>>`
    }
}

fn main() {}
