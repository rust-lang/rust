//@ edition:2024

#![feature(gen_blocks)]

const gen fn a() {}
//~^ ERROR functions cannot be both `const` and `gen`

const async gen fn b() {}
//~^ ERROR functions cannot be both `const` and `async gen`

fn main() {}
