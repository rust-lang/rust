//! Test that `Rc<T>` cannot be sent between threads.

use std::rc::Rc;
use std::thread;

#[derive(Debug)]
struct Port<T>(Rc<T>);

#[derive(Debug)]
struct Foo {
    _x: Port<()>,
}

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn foo(x: Port<()>) -> Foo {
    Foo { _x: x }
}

fn main() {
    let x = foo(Port(Rc::new(())));

    thread::spawn(move || {
        //~^ ERROR `Rc<()>` cannot be sent between threads safely
        let y = x;
        println!("{:?}", y);
    });
}
