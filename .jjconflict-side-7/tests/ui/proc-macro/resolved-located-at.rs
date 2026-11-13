//@ proc-macro: resolved-located-at.rs

#[macro_use]
extern crate resolved_located_at;

fn main() {
    resolve_located_at!(a b)
    //~^ ERROR expected error
    //~| ERROR mismatched types
}
