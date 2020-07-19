// check-pass
use std::rc::Rc;

trait A {
    fn foo(&self) {}
}

impl<T: Send> A for T {}

fn test<T: A>(rc: &Rc<T>) {
    rc.foo()
    // `Rc: Send` must not be evaluated as ambiguous
    // for this to compile, as we are otherwise
    // not allowed to use auto deref here to use
    // the `T: A` implementation.
}

fn main() {}
