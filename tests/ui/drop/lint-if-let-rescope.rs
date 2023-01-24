//@ edition:2024
//@ compile-flags: -Z validate-mir -Zunstable-options
//@ run-rustfix

#![feature(if_let_rescope)]
#![deny(if_let_rescope)]
#![allow(irrefutable_let_patterns)]

fn droppy() -> Droppy {
    Droppy
}
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
    if let Some(_value) = droppy().get() {
        //~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //~| WARN: this changes meaning in Rust 2024
        //~| HELP: rewrite this `if let` into a `match`
        // do something
    } else {
        //~^ HELP: the value is now dropped here in Edition 2024
        // do something else
    }

    if let Some(1) = { if let Some(_value) = Droppy.get() { Some(1) } else { None } } {
        //~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //~| WARN: this changes meaning in Rust 2024
        //~| HELP: rewrite this `if let` into a `match`
        //~| HELP: the value is now dropped here in Edition 2024
    }

    if let () = { if let Some(_value) = Droppy.get() {} } {
        //~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //~| WARN: this changes meaning in Rust 2024
        //~| HELP: rewrite this `if let` into a `match`
        //~| HELP: the value is now dropped here in Edition 2024
    }
}
