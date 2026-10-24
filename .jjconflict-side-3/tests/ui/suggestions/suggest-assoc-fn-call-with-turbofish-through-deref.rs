use std::cell::RefCell;

struct HasAssocMethod;

impl HasAssocMethod {
    fn hello() {}
}
fn main() {
    let shared_state = RefCell::new(HasAssocMethod);
    let state = shared_state.borrow_mut();
    state.hello();
    //~^ ERROR no method named `hello` found
}
