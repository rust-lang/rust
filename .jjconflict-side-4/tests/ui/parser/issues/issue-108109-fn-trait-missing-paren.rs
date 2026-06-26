//@ run-rustfix

pub fn func<F>() where F: FnOnce -> () {}
//~^ ERROR expected one of
//~| NOTE expected one of
//~| NOTE `Fn` bounds require arguments in parentheses

fn main() {}
