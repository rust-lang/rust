#![allow(unused)]

const fn f<T>(x: T) { //~ WARN function cannot return without recursing
    f(x);
    //~^ ERROR evaluation of constant value failed
}

const X: () = f(1);

fn main() {}
