//@ aux-build: transitive-dep.rs
//@ aux-build: direct-dep.rs

extern crate direct_dep;

struct Struct;
//~^ NOTE `Struct` is defined in the current crate

fn main() {
    let _: direct_dep::__private::Struct = Struct;
    //~^ ERROR mismatched types
    //~| NOTE expected `direct_dep::__private::Struct`, found `Struct`
    //~| NOTE expected due to this
    //~| NOTE `Struct` and `direct_dep::__private::Struct` have similar names, but are actually distinct types
    //~| NOTE `direct_dep::__private::Struct` is defined in crate `transitive_dep`
}
