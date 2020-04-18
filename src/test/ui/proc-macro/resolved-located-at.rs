// aux-build:resolved-located-at.rs

#![feature(proc_macro_hygiene)]

#[macro_use]
extern crate resolved_located_at;

fn main() {
    resolve_located_at!(a b)
    //~^ ERROR expected error
    //~| ERROR mismatched types
}
