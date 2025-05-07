#![allow(unused)]

const fn f<T>(x: T) { //~ WARN function cannot return without recursing
    f(x);
}

const X: () = f(1); //~ ERROR evaluation of constant value failed

fn main() {}
