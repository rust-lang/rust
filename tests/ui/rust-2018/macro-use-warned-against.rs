//@ aux-build:macro-use-warned-against.rs
//@ aux-build:macro-use-warned-against2.rs
//@ check-pass

#![warn(macro_use_extern_crate, unused)]

#[macro_use] //~ WARN applying the `#[macro_use]` attribute to an `extern crate` item is deprecated
extern crate macro_use_warned_against;
#[macro_use] //~ WARN unused `#[macro_use]`
extern crate macro_use_warned_against2;

fn main() {
    foo!();
}
