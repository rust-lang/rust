#![feature(plugin)]
#![plugin(clippy)]

#![deny(clippy)]
#![allow(unused_assignments)]

fn main() {
    let mut a = 42;
    let mut b = 1337;

    a = b;
    b = a;
    //~^^ ERROR this looks like you are trying to swap `a` and `b`
    //~| HELP try
    //~| SUGGESTION std::mem::swap(a, b);
    //~| NOTE or maybe you should use `std::mem::replace`?
}
