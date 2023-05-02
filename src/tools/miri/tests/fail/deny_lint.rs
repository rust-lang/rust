//@error-pattern: miri cannot be run on programs that fail compilation

#![deny(warnings)]

struct Foo;
//~^ ERROR: struct `Foo` is never constructed

fn main() {}
