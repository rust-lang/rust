// edition: 2024
// compile-flags: -Zunstable-options
#![feature(gen_blocks)]

// async generators are not yet supported, so this test makes sure they make some kind of reasonable
// error.

async gen fn foo() {}
//~^ `async gen` functions are not supported

fn main() {}
