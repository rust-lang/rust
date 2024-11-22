//@ edition: 2024
//@ check-pass

#![feature(gen_blocks)]

fn diverge() -> ! { loop {} }

async gen fn async_gen_fn() -> i32 { diverge() }

gen fn gen_fn() -> i32 { diverge() }

fn async_gen_block() {
    async gen { yield (); diverge() };
}

fn gen_block() {
    gen { yield (); diverge() };
}

fn main() {}
