use std::fmt;

struct S {
}

impl S {
    fn hello<P>(&self, val: &P) where P: fmt::Display; {
        //~^ ERROR non-item in item list
        //~| ERROR associated function in `impl` without body
        println!("val: {}", val);
    }
}

impl S {
    fn hello_empty<P>(&self, val: &P) where P: fmt::Display;
    //~^ ERROR associated function in `impl` without body
}

fn main() {
    let s = S{};
    s.hello(&32);
}
