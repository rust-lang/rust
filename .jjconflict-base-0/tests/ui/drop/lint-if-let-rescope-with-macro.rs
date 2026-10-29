// This test ensures that no suggestion is emitted if the span originates from
// an expansion that is probably not under a user's control.

//@ edition:2021
//@ compile-flags: -Z unstable-options

#![deny(if_let_rescope)]
#![allow(irrefutable_let_patterns)]

macro_rules! edition_2021_if_let {
    ($p:pat, $e:expr, { $($conseq:tt)* } { $($alt:tt)* }) => {
        if let $p = $e { $($conseq)* } else { $($alt)* }
        //~^ ERROR `if let` assigns a shorter lifetime since Edition 2024
        //~| WARN this changes meaning in Rust 2024
    }
}

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
    edition_2021_if_let! {
        Some(_value),
        droppy().get(),
        {}
        {}
    };
}
