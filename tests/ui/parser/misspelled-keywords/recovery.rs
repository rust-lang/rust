// Shows that we perform recovery on misspelled item keyword.

#![feature(associated_type_defaults)]

trait Animal {
    Type Result = u8;
    //~^ ERROR keyword `type` is written in the wrong case
}

Struct Foor {
    //~^ ERROR keyword `struct` is written in the wrong case
    hello: String,
}

Const A: u8 = 10;
//~^ ERROR keyword `const` is written in the wrong case

Fn code() {}
//~^ ERROR keyword `fn` is written in the wrong case

Static a: u8 = 0;
//~^ ERROR keyword `static` is written in the wrong case

usee a::b;
//~^ ERROR expected one of

fn main() {}
