//@ edition:2024

#![feature(gen_blocks)]

const gen fn a() {}
//~^ ERROR functions cannot be both `const` and `gen`
//~^^ ERROR `gen` fn bodies are not allowed in constant functions

const async gen fn b() {}
//~^ ERROR functions cannot be both `const` and `async gen`
//~^^ ERROR `async gen` fn bodies are not allowed in constant functions

fn main() {}
