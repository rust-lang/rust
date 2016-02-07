#![feature(plugin)]
#![plugin(clippy)]
#![deny(print_stdout, use_debug)]

use std::fmt::{Debug, Display, Formatter, Result};

#[allow(dead_code)]
struct Foo;

impl Display for Foo {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{:?}", 43.1415)
        //~^ ERROR use of `Debug`-based formatting
    }
}

impl Debug for Foo {
    fn fmt(&self, f: &mut Formatter) -> Result {
        // ok, we can use `Debug` formatting in `Debug` implementations
        write!(f, "{:?}", 42.718)
    }
}

fn main() {
    println!("Hello"); //~ERROR use of `println!`
    print!("Hello"); //~ERROR use of `print!`

    print!("Hello {}", "World"); //~ERROR use of `print!`

    print!("Hello {:?}", "World");
    //~^ ERROR use of `print!`
    //~| ERROR use of `Debug`-based formatting

    print!("Hello {:#?}", "#orld");
    //~^ ERROR use of `print!`
    //~| ERROR use of `Debug`-based formatting

    assert_eq!(42, 1337);

    vec![1, 2];
}
