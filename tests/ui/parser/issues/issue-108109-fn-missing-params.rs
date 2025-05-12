//@ run-rustfix

pub fn missing -> () {}
//~^ ERROR missing parameters for function definition

pub fn missing2 {}
//~^ ERROR missing parameters for function definition

fn main() {}
