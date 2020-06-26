#![feature(const_loop)]

static _X: () = loop {}; //~ ERROR could not evaluate static initializer

fn main() {}
