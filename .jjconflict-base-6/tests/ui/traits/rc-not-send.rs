//! Test that `Rc<T>` does not implement `Send`.

use std::rc::Rc;

fn requires_send<T: Send>(_: T) {}

fn main() {
    let rc_value = Rc::new(5);
    requires_send(rc_value);
    //~^ ERROR `Rc<{integer}>` cannot be sent between threads safely
}
