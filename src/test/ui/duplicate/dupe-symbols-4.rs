//
// error-pattern: symbol `fail` is already defined
#![crate_type="rlib"]
#![allow(warnings)]


pub trait A {
    fn fail(self);
}

struct B;
struct C;

impl A for B {
    #[no_mangle]
    fn fail(self) {}
}

impl A for C {
    #[no_mangle]
    fn fail(self) {}
}
