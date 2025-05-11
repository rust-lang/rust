use std::cell::Cell;
use std::rc::Rc;

fn send<T: Send>(_: T) {}

fn main() {}

// Cycles should work as the deferred obligations are
// independently resolved and only require the concrete
// return type, which can't depend on the obligation.
fn cycle1() -> impl Clone {
    send(cycle2().clone());

    Rc::new(Cell::new(5))
}

fn cycle2() -> impl Clone {
    send(cycle1().clone());
    //~^ ERROR: cannot check whether the hidden type of opaque type satisfies auto traits

    Rc::new(String::from("foo"))
}
