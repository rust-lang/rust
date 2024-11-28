// This test checks that the lint `if_let_rescope` only actions
// when Edition 2021 or prior is targeted here because the lint should work especially
// when edition migration towards 2024 is executed.

//@ revisions: edition2021 edition2024
//@ [edition2021] edition: 2021
//@ [edition2024] edition: 2024
//@ [edition2024] check-pass

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
        //[edition2021]~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //[edition2021]~| HELP: a `match` with a single arm can preserve the drop order up to Edition 2021
        //[edition2021]~| WARN: this changes meaning in Rust 2024
    } else {
        //[edition2021]~^ HELP: the value is now dropped here in Edition 2024
    }
}
