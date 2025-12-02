// Shows that we perform recovery on misspelled item keyword.

#![feature(associated_type_defaults)]

trait Animal {
    Type Result = u8;
    //~^ ERROR expected one of
}

Struct Foor {
    //~^ ERROR expected one of
    hello: String,
}

Const A: u8 = 10;

Fn code() {}

Static a: u8 = 0;

usee a::b;

fn main() {}
