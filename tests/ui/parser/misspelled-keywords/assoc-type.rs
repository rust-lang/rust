#![feature(associated_type_defaults)]

trait Animal {
    Type Result = u8;
    //~^ ERROR keyword `type` is written in the wrong case
}

fn main() {}
