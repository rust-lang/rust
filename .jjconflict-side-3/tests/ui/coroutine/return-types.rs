//@ edition: 2024

#![feature(gen_blocks)]

async gen fn async_gen_fn() -> i32 { 0 }
//~^ ERROR mismatched types

gen fn gen_fn() -> i32 { 0 }
//~^ ERROR mismatched types

fn async_gen_block() {
    async gen { yield (); 1 };
    //~^ ERROR mismatched types
}

fn gen_block() {
    gen { yield (); 1 };
    //~^ ERROR mismatched types
}

fn main() {}
