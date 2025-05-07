#![allow(clippy::print_literal, clippy::write_literal)]
#![warn(clippy::print_stdout, clippy::use_debug)]

use std::fmt::{Debug, Display, Formatter, Result};

#[allow(dead_code)]
struct Foo;

impl Display for Foo {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{:?}", 43.1415)
        //~^ use_debug
    }
}

impl Debug for Foo {
    fn fmt(&self, f: &mut Formatter) -> Result {
        // ok, we can use `Debug` formatting in `Debug` implementations
        write!(f, "{:?}", 42.718)
    }
}

fn main() {
    println!("Hello");
    //~^ print_stdout

    print!("Hello");
    //~^ print_stdout

    print!("Hello {}", "World");
    //~^ print_stdout

    print!("Hello {:?}", "World");
    //~^ print_stdout
    //~| use_debug

    print!("Hello {:#?}", "#orld");
    //~^ print_stdout
    //~| use_debug

    assert_eq!(42, 1337);

    vec![1, 2];
}
