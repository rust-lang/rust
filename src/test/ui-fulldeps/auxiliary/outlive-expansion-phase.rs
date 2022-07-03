// force-host

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_middle;

use rustc_driver::plugin::Registry;
use std::any::Any;
use std::cell::RefCell;

struct Foo {
    foo: isize,
}

impl Drop for Foo {
    fn drop(&mut self) {}
}

#[no_mangle]
fn __rustc_plugin_registrar(_: &mut Registry) {
    thread_local!(static FOO: RefCell<Option<Box<Any+Send>>> = RefCell::new(None));
    FOO.with(|s| *s.borrow_mut() = Some(Box::new(Foo { foo: 10 }) as Box<Any + Send>));
}
