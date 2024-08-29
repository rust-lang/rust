#![crate_type = "rlib"]

extern crate foo;

pub fn bar() {
    foo::foo()
}
