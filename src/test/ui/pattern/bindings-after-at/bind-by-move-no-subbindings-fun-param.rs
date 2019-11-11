// See issue #12534.

#![feature(bindings_after_at)]
//~^ WARN the feature `bindings_after_at` is incomplete and may cause the compiler to crash

fn main() {}

struct A(Box<u8>);

fn f(a @ A(u): A) -> Box<u8> {
    //~^ ERROR cannot bind by-move with sub-bindings
    //~| ERROR use of moved value
    drop(a);
    u
}
