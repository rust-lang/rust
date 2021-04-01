extern "wasm" fn foo() {
    //~^ ERROR: wasm ABI is experimental and subject to change
}

fn main() {
    foo();
}
