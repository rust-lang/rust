use std::cell::RefCell;

pub struct MessageListeners<'a> {
    listeners: RefCell<Vec<Box<dyn FnMut(()) + 'a>>>,
}

pub trait MessageListenersInterface {
    fn listeners<'c>(&'c self) -> &'c MessageListeners<'c>;
}

impl<'a> MessageListenersInterface for MessageListeners<'a> {
    fn listeners<'b>(&'b self) -> &'a MessageListeners<'b> {
        //~^ ERROR in type `&'a MessageListeners<'_>`, reference has a longer lifetime than the data it references
        self
    }
}

fn main() {}
