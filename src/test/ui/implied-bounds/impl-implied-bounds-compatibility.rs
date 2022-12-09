use std::cell::RefCell;

pub struct MessageListeners<'a> {
    listeners: RefCell<Vec<Box<dyn FnMut(()) + 'a>>>,
}

pub trait MessageListenersInterface {
    fn listeners<'c>(&'c self) -> &'c MessageListeners<'c>;
}

impl<'a> MessageListenersInterface for MessageListeners<'a> {
    fn listeners<'b>(&'b self) -> &'a MessageListeners<'b> {
        //~^ ERROR cannot infer an appropriate lifetime for lifetime parameter 'b in generic type due to conflicting requirement
        self
    }
}

fn main() {}
