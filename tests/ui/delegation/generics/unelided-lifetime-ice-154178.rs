//@ compile-flags: -Z deduplicate-diagnostics=yes

#![feature(fn_delegation)]

fn foo<'b: 'b, const N: usize>() {}

trait Trait {
    reuse foo::<1>;
    //~^ ERROR: function takes 1 lifetime argument but 0 lifetime arguments were supplied
}

fn main() {}
