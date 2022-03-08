#![allow(unused)]

const fn f<T>(x: T) { //~ WARN function cannot return without recursing
    f(x);
    //~^ ERROR any use of this value will cause an error
    //~| WARN this was previously accepted by the compiler
}

const X: () = f(1);

fn main() {}
