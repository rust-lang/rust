// This test checks that the lint `if_let_rescope` only actions
// when the feature gate is enabled.
// Edition 2021 is used here because the lint should work especially
// when edition migration towards 2024 is run.

//@ edition: 2021

#![deny(if_let_rescope)]
#![allow(irrefutable_let_patterns)]

struct Droppy;
impl Drop for Droppy {
    fn drop(&mut self) {
        println!("dropped");
    }
}
impl Droppy {
    fn get(&self) -> Option<u8> {
        None
    }
}

fn main() {
    if let Some(_value) = Droppy.get() {
        //~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //~| HELP: a `match` with a single arm can preserve the drop order up to Edition 2021
        //~| WARN: this changes meaning in Rust 2024
    } else {
        //~^ HELP: the value is now dropped here in Edition 2024
    }
}
