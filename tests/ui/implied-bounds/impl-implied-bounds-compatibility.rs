#![deny(implied_bounds_entailment)]

use std::cell::RefCell;

pub struct MessageListeners<'a> {
    listeners: RefCell<Vec<Box<dyn FnMut(()) + 'a>>>,
}

pub trait MessageListenersInterface {
    fn listeners<'c>(&'c self) -> &'c MessageListeners<'c>;
}

impl<'a> MessageListenersInterface for MessageListeners<'a> {
    fn listeners<'b>(&'b self) -> &'a MessageListeners<'b> {
        //~^ ERROR impl method assumes more implied bounds than the corresponding trait method
        //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        self
    }
}

fn main() {}
