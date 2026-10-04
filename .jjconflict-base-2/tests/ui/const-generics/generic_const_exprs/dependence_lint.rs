//@ revisions: full gce
//@ compile-flags: -Zdeduplicate-diagnostics=yes

#![cfg_attr(gce, feature(generic_const_exprs))]
#![allow(incomplete_features)]

use std::mem::size_of;

fn foo<T>() {
    [0; size_of::<*mut T>()]; // lint on stable, error with `generic_const_exprs`
    //[gce]~^ ERROR unconstrained
    //[gce]~| ERROR unconstrained generic constant
    //[full]~^^^ WARNING cannot use constants
    //[full]~| WARNING this was previously accepted
    let _: [u8; size_of::<*mut T>()]; // error on stable, error with gce
    //[full]~^ ERROR generic parameters may not be used
    //[gce]~^^ ERROR unconstrained generic
    [0; if false { size_of::<T>() } else { 3 }]; // lint on stable, error with gce
    //[gce]~^ ERROR overly complex
    //[full]~^^ WARNING cannot use constants
    //[full]~| WARNING this was previously accepted
    let _: [u8; if true { size_of::<T>() } else { 3 }]; // error on stable, error with gce
    //[full]~^ ERROR generic parameters may not be used
    //[gce]~^^ ERROR overly complex
}

fn main() {}
