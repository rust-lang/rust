//@ compile-flags: -Zwrite-long-types-to-disk=yes
use std::cell::Cell;
use std::rc::Rc;

// Fast path, main can see the concrete type returned.
fn before() -> impl Fn(i32) {
//~^ NOTE within this `impl Fn
//~| NOTE within the type `impl Fn
//~| NOTE expansion of desugaring
    let p = Rc::new(Cell::new(0));
    move |x| p.set(x) //~ NOTE used within this closure
}

fn send<T: Send>(_: T) {}
//~^ NOTE required by a bound
//~| NOTE required by a bound
//~| NOTE required by this bound
//~| NOTE required by this bound

fn main() {
    send(before());
    //~^ ERROR `Rc<Cell<i32>>` cannot be sent between threads safely
    //~| NOTE `Rc<Cell<i32>>` cannot be sent between threads safely
    //~| NOTE required by a bound

    send(after());
    //~^ ERROR `Rc<Cell<i32>>` cannot be sent between threads safely
    //~| NOTE `Rc<Cell<i32>>` cannot be sent between threads safely
    //~| NOTE required by a bound
}

// Deferred path, main has to wait until typeck finishes,
// to check if the return type of after is Send.
fn after() -> impl Fn(i32) {
//~^ NOTE within this `impl Fn(i32)`
//~| NOTE in this expansion
//~| NOTE appears within the type
    let p = Rc::new(Cell::new(0));
    move |x| p.set(x) //~ NOTE used within this closure
}
