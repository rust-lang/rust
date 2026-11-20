//@ edition: 2024

#![feature(gen_blocks)]

async fn async_fn() {
    break; //~ ERROR `break` inside `async` function
}

gen fn gen_fn() {
    break; //~ ERROR `break` inside `gen` function
}

async gen fn async_gen_fn() {
    break; //~ ERROR `break` inside `async gen` function
}

fn main() {
    let _ = async { break; }; //~ ERROR `break` inside `async` block

    let _ = async || { break; }; //~ ERROR `break` inside `async` closure

    let _ = gen { break; }; //~ ERROR `break` inside `gen` block

    let _ = async gen { break; }; //~ ERROR `break` inside `async gen` block
}
