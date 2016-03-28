#![feature(plugin)]
#![plugin(clippy)]

#![deny(clippy)]
#![allow(unused_assignments)]

struct Foo(u32);

fn main() {
    let mut a = 42;
    let mut b = 1337;

    a = b;
    b = a;
    //~^^ ERROR this looks like you are trying to swap `a` and `b`
    //~| HELP try
    //~| SUGGESTION std::mem::swap(&mut a, &mut b);
    //~| NOTE or maybe you should use `std::mem::replace`?

    let t = a;
    a = b;
    b = t;
    //~^^^ ERROR this looks like you are swapping `a` and `b` manually
    //~| HELP try
    //~| SUGGESTION std::mem::swap(&mut a, &mut b);
    //~| NOTE or maybe you should use `std::mem::replace`?

    let mut c = Foo(42);

    c.0 = a;
    a = c.0;
    //~^^ ERROR this looks like you are trying to swap `c.0` and `a`
    //~| HELP try
    //~| SUGGESTION std::mem::swap(&mut c.0, &mut a);
    //~| NOTE or maybe you should use `std::mem::replace`?

    let t = c.0;
    c.0 = a;
    a = t;
    //~^^^ ERROR this looks like you are swapping `c.0` and `a` manually
    //~| HELP try
    //~| SUGGESTION std::mem::swap(&mut c.0, &mut a);
    //~| NOTE or maybe you should use `std::mem::replace`?
}
