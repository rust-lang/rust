//@ edition: 2021
//@ normalize-stderr: "\n\n\z" -> "\n"

#![allow(unused)]
#![deny(unwind_drop_order)]

use std::fmt::Display;

struct Wrap<T: Display>(T);

impl<T: Display> Drop for Wrap<T> {
    fn drop(&mut self) {
        println!("{}", self.0);
    }
}

fn main() {
    let x;
    {
        let y = 1;
        x = Wrap(&y);
        //~^ ERROR relative drop order on unwind changing in a future release
        //~| WARNING this will change its meaning in a future release
        panic!();
    }
}
