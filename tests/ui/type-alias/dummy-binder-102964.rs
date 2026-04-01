//! Regression test for https://github.com/rust-lang/rust/issues/102964

use std::rc::Rc;
type Foo<'a, T> = &'a dyn Fn(&T);
type RcFoo<'a, T> = Rc<Foo<'a, T>>;

fn bar_function<T>(function: Foo<T>) -> RcFoo<T> {
    //~^ ERROR mismatched types
    let rc = Rc::new(function);
}

fn main() {}
