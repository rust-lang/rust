#![feature(const_fn)]

fn main() {}

const fn foo() {}
const X: fn() = foo;

const fn bar() {
    X()
    //~^ ERROR function pointers in const fn are unstable
}
