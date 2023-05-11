use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Trader<'a> {
    closure: Box<dyn Fn(&mut Trader) + 'a>,
}

impl<'a> Trader<'a> {
    pub fn new() -> Self {
        Trader {
            closure: Box::new(|_| {}),
        }
    }
    pub fn set_closure(&mut self, function: impl Fn(&mut Trader) + 'a) {
        //foo
    }
}

fn main() {
    let closure = |trader : Trader| {
        println!("Woooosh!");
    };

    let mut trader = Trader::new();
    trader.set_closure(closure);
    //~^ ERROR type mismatch in closure arguments
}
