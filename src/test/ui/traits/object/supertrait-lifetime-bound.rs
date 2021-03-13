// check-pass

use std::any::Any;

trait A<T>: Any {
    fn m(&self) {}
}

impl<S, T: 'static> A<S> for T {}

fn call_obj<'a>() {
    let obj: &dyn A<&'a ()> = &();
    obj.m();
}

fn main() {}
