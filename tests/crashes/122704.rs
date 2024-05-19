//@ known-bug: #122704
use std::any::Any;

pub struct Foo {
    bar: Box<dyn for<'a> Fn(&'a usize) -> Box<dyn Any + 'a>>,
}

impl Foo {
    pub fn ack<I>(&mut self, f: impl for<'a> Fn(&'a usize) -> Box<I>) {
        self.bar = Box::new(|baz| Box::new(f(baz)));
    }
}

fn main() {}
