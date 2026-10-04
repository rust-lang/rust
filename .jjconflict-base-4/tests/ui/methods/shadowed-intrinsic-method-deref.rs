//@ run-rustfix
#![allow(unused_imports)]
use std::rc::Rc;
use std::cell::RefCell;
use std::borrow::Borrow; // Without this import, the code would compile.

pub struct S {
    flag: bool,
}

type SCell = Rc<RefCell<S>>;

fn main() {
    // Type annotations just for clarity
    let s : SCell = Rc::new(RefCell::new(S {flag: false}));
    let sb : &S = &s.borrow();
    //~^ ERROR: the trait bound `Rc<RefCell<S>>: Borrow<S>` is not satisfied [E0277]
    //~| NOTE: the trait `Borrow<S>` is not implemented for `Rc<RefCell<S>>`
    //~| NOTE: there's an inherent method on `RefCell<S>` of the same name
    println!("{:?}", sb.flag);
}
