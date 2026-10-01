// Regression test for #119783

#![expect(incomplete_features)]
#![feature(gca_min_const_items)]

trait Trait {
    #[rustc_always_gca]
    const F: fn();
    //~^ ERROR using function pointers as const generic parameters is forbidden
}

fn take(_: impl Trait<F = const { || {} }>) {}

fn main() {}
