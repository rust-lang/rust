// Test handling of unstable items dependent on multiple features.
//@ aux-build:two-unstables.rs
//@ revisions: all some none
//@ [all]check-pass

#![cfg_attr(all, feature(a, b, c, d, e, f, g, h))]
#![cfg_attr(some, feature(a, d, e, h))]

extern crate two_unstables;

const USE_NOTHING: () = two_unstables::nothing();
//[none]~^ ERROR `nothing` is not yet stable as a const fn
//[some]~^^ ERROR `nothing` is not yet stable as a const fn

struct Wrapper(two_unstables::Foo);
//[none]~^ ERROR use of unstable library features `a` and `b`: reason [E0658]
//[some]~^^ ERROR use of unstable library feature `b`: reason [E0658]

impl two_unstables::Trait for Wrapper {}
//[none]~^ ERROR not all trait items implemented, missing: `method` [E0046]
//[some]~^^ ERROR not all trait items implemented, missing: `method` [E0046]

fn main() {
    two_unstables::mac!();
    //[none]~^ ERROR use of unstable library features `g` and `h`: reason [E0658]
    //[some]~^^ ERROR use of unstable library feature `g`: reason [E0658]
}
