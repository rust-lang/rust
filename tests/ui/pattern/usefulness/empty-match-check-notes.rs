//@ aux-build:empty.rs
//@ revisions: normal exhaustive_patterns
//
// This tests a match with no arms on various types, and checks NOTEs.
#![feature(never_type)]
#![cfg_attr(exhaustive_patterns, feature(exhaustive_patterns))]
#![deny(unreachable_patterns)]
//~^ NOTE the lint level is defined here

extern crate empty;

enum EmptyEnum {}

fn empty_enum(x: EmptyEnum) {
    match x {} // ok
    match x {
        _ => {} //~ ERROR unreachable pattern
                //~^ NOTE matches no values
                //~| NOTE to learn more about uninhabited types, see
    }
    match x {
        _ if false => {} //~ ERROR unreachable pattern
                         //~^ NOTE matches no values
                         //~| NOTE to learn more about uninhabited types, see
    }
}

fn empty_foreign_enum(x: empty::EmptyForeignEnum) {
    match x {} // ok
    match x {
        _ => {} //~ ERROR unreachable pattern
                //~^ NOTE matches no values
                //~| NOTE to learn more about uninhabited types, see
    }
    match x {
        _ if false => {} //~ ERROR unreachable pattern
                         //~^ NOTE matches no values
                         //~| NOTE to learn more about uninhabited types, see
    }
}

fn empty_foreign_enum_private(x: &Option<empty::SecretlyUninhabitedForeignStruct>) {
    let None = *x;
    //~^ ERROR refutable pattern in local binding
    //~| NOTE `let` bindings require an "irrefutable pattern"
    //~| NOTE for more information, visit
    //~| NOTE the matched value is of type
    //~| NOTE pattern `Some(_)` not covered
    //~| NOTE currently uninhabited, but this variant contains private fields
}

fn main() {
    match 0u8 {
        //~^ ERROR not covered
        //~| NOTE the matched value is of type
        //~| NOTE match arms with guards don't count towards exhaustivity
        //~| NOTE not covered
        _ if false => {}
    }
}
