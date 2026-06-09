// Regression test for #122704
use std::any::Any;

pub struct Foo {
    bar: Box<dyn for<'a> Fn(&'a usize) -> Box<dyn Any + 'a>>,
}

impl Foo {
    pub fn ack<I>(&mut self, f: impl for<'a> Fn(&'a usize) -> Box<I>) {
        self.bar = Box::new(|baz| Box::new(f(baz)));
        //~^ ERROR the parameter type `impl for<'a> Fn(&'a usize) -> Box<I>` may not live long enough
        //~| ERROR the parameter type `impl for<'a> Fn(&'a usize) -> Box<I>` may not live long enough
        //~| ERROR the parameter type `impl for<'a> Fn(&'a usize) -> Box<I>` may not live long enough
        //~| ERROR the parameter type `impl for<'a> Fn(&'a usize) -> Box<I>` may not live long enough
        //~| ERROR the parameter type `I` may not live long enough
        //~| ERROR the parameter type `I` may not live long enough
        //~| ERROR the parameter type `I` may not live long enough
        //~| ERROR `f` does not live long enough
    }
}

fn main() {}
