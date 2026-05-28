//@ run-pass
#![allow(dead_code)]

trait Hax {
    fn dummy(&self) {}
}
impl<A> Hax for A {}

fn perform_hax<T: 'static>(x: Box<T>) -> Box<dyn Hax + 'static> {
    Box::new(x) as Box<dyn Hax + 'static>
}

fn deadcode() {
    perform_hax(Box::new("deadcode".to_string()));
}

pub fn main() {
    perform_hax(Box::new(42));
}
