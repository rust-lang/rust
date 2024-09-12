// This test checks that the lint `if_let_rescope` only actions
// when the feature gate is enabled.
// Edition 2021 is used here because the lint should work especially
// when edition migration towards 2024 is run.

//@ revisions: with_feature_gate without_feature_gate
//@ [without_feature_gate] check-pass
//@ edition: 2021

#![cfg_attr(with_feature_gate, feature(if_let_rescope))]
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
        //[with_feature_gate]~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //[with_feature_gate]~| HELP: a `match` with a single arm can preserve the drop order up to Edition 2021
        //[with_feature_gate]~| WARN: this changes meaning in Rust 2024
    } else {
        //[with_feature_gate]~^ HELP: the value is now dropped here in Edition 2024
    }
}
