//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ edition: 2024

//! Regression test for https://github.com/rust-lang/rust/issues/146016.
//! Locate the field that introduces an auto-trait requirement, even through a type alias.

use std::sync::mpsc;
use std::thread;

mod other {
    use std::ffi::c_void;

    pub type Bar = *const c_void;
}

use other::Bar;

enum Foo {
    Case(i32),
    Case2(Bar),
}

struct Record<T> {
    unrelated: u8,
    value: T,
}

struct Nested<T> {
    value: Record<T>,
}

enum Multiple<T> {
    Empty,
    Values { first: T, second: T },
}

fn require_send<T: Send>() {}

fn generic_fields() {
    require_send::<Record<Bar>>();
    //~^ ERROR `*const c_void` cannot be sent between threads safely
    require_send::<Nested<Bar>>();
    //~^ ERROR `*const c_void` cannot be sent between threads safely
    require_send::<Multiple<Bar>>();
    //~^ ERROR `*const c_void` cannot be sent between threads safely

    require_send::<Record<u8>>();
    require_send::<Nested<u8>>();
    require_send::<Multiple<u8>>();
}

fn main() {
    let (tx, rx) = mpsc::channel();
    let h = thread::spawn(move || match rx.recv().unwrap() {
        //~^ ERROR `*const c_void` cannot be sent between threads safely
        Foo::Case(x) => println!("{:?}", x),
        Foo::Case2(b) => println!("{:?}", b),
    });
    tx.send(Foo::Case2(std::ptr::null()));
    h.join();
}
