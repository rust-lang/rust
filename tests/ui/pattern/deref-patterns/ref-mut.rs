#![feature(deref_patterns)]
//~^ WARN the feature `deref_patterns` is incomplete

use std::rc::Rc;

fn main() {
    match &mut vec![1] {
        deref!(x) => {}
        _ => {}
    }
    match &mut vec![1] {
        [x] => {}
        _ => {}
    }

    match &mut Rc::new(1) {
        deref!(x) => {}
        //~^ ERROR the trait bound `Rc<{integer}>: DerefMut` is not satisfied
        _ => {}
    }
    match &mut Rc::new((1,)) {
        (x,) => {}
        //~^ ERROR the trait bound `Rc<({integer},)>: DerefMut` is not satisfied
        _ => {}
    }
}
