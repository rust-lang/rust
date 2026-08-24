//@ aux-build: transitive-dep.rs
//@ aux-build: direct-dep-with-multiple-reexports.rs

extern crate direct_dep_with_multiple_reexports as direct_dep;

struct Struct;
//~^ NOTE `Struct` is defined in the current crate

fn main() {
    let _: direct_dep::Struct = Struct;
    //~^ ERROR mismatched types
    //~| NOTE expected `direct_dep::Struct`, found `Struct`
    //~| NOTE expected due to this
    //~| NOTE `Struct` and `direct_dep::Struct` have similar names, but are actually distinct types
    //~| NOTE `direct_dep::Struct` is defined in crate `transitive_dep`
}
