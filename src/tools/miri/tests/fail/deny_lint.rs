//@error-in-other-file: miri cannot be run on programs that fail compilation

#![deny(warnings, unused)]

struct Foo;
//~^ ERROR: struct `Foo` is never constructed

fn main() {}
